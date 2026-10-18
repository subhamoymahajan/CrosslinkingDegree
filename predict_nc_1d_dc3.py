"""Predict NC_1D data and compare DC3/DC1 with measured salt properties.

Run: conda run -n tfml python predict_nc_1d_dc3.py
Uses elemental O/N and f with the two-input CV fold having lowest held-out RMSE.
Correlations use shared finite rows for each property, independently.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dc3_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import tensorflow as tf


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'neural_network_cross_validation/nc_1d_predictions'
COLORS = {'DC3': hsv_to_rgb((.58, 1., .8)), 'DC1': hsv_to_rgb((.58, .7, .8)),
          'HABD': hsv_to_rgb((.58, .4, .8))}
YLABELS = {'R': r'$R$ (%)', 'F': r'$F$ (L m$^{-2}$ h$^{-1}$ bar$^{-1}$)',
           'B': r'$B$ (L m$^{-2}$ h$^{-1}$)'}
LABELS = {'NaCl': 'NaCl', 'Na2SO4': r'Na$_2$SO$_4$'}


def save(fig, stem):
    for ext in ['png']:
        fig.savefig(stem.with_suffix('.' + ext), dpi=300)
    plt.close(fig)


def scatter(pairs, salt, metric, outcome, r, output_dir=OUT):
    folder = output_dir / 'property_vs_DC3'
    folder.mkdir(exist_ok=True)
    stem = folder / f'{salt}_{metric}_vs_DC3_f1'
    pairs.to_csv(stem.with_suffix('.csv'), index=False)
    fig, ax = plt.subplots(figsize=(3, 3), layout='constrained')
    ax.scatter(pairs.DC3, pairs[outcome], s=22, facecolors='none',
               edgecolors=[COLORS['DC3']], linewidths=.7)
    ax.scatter(pairs.DC3, pairs[outcome], s=22, facecolors=[COLORS['DC3']],
               edgecolors='none', alpha=.5)
    ax.set(xlabel=r'Predicted $DC_3$ ($f=1$)', ylabel=YLABELS[metric], xlim=(0, 1))
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.margins(y=.18)
    ax.text(.97, .97, f'{LABELS[salt]}\n$r={r:.3f}$, $n={len(pairs)}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)
    save(fig, stem)


def bars(results, method, output_dir=OUT, descriptors=('DC3', 'DC1')):
    fig, ax = plt.subplots(figsize=(3, 3))
    fig.subplots_adjust(left=.24, right=.98, bottom=.18, top=.78)
    salt = 'NaCl'
    data = results.loc[(results.dataset == salt) & np.isclose(results.bond_fraction, 1)
                       & (results.method == method)]
    if data.empty:
        plt.close(fig)
        return
    x = np.arange(3)
    width = .6 / len(descriptors)
    legend_labels = {'DC3': r'$DC_3$ ($f=1$)', 'DC1': r'$DC_1$', 'HABD': 'HABD'}
    for index, descriptor in enumerate(descriptors):
        offset = (index - (len(descriptors) - 1) / 2) * width
        selected = data.loc[data.descriptor == descriptor].set_index('metric').loc[['R', 'F', 'B']]
        values = selected.correlation.abs()
        ax.bar(x + offset, values, width=width, color=COLORS[descriptor],
               edgecolor='.2', linewidth=.6,
               label=legend_labels[descriptor])
        for pos, value in zip(x + offset, values):
            ax.annotate(f'{value:.2f}', (pos, value), xytext=(0, 3),
                        textcoords='offset points', ha='center', fontsize=6 if len(descriptors) > 2 else 7)
    counts = data.loc[data.descriptor == 'DC3'].set_index('metric').loc[['R', 'F', 'B'], 'n']
    ax.set_xticks(x, [f'{m}\n$n={n}$' for m, n in zip(['R', 'F', 'B'], counts)])
    ax.text(.5, .97, LABELS[salt], transform=ax.transAxes, ha='center', va='top')
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.set_ylabel(r'$|\mathrm{Pearson}\ r|$' if method == 'pearson'
                       else r'$|\mathrm{Spearman}\ \rho|$')
    handles, labels = ax.get_legend_handles_labels()
    legend_options = dict(fontsize=8, columnspacing=1, handlelength=1.5) if len(descriptors) > 2 else {}
    fig.legend(handles, labels, loc='upper center', ncol=len(descriptors), frameon=False, **legend_options)
    name = 'combined_correlations_f1' if method == 'pearson' else 'combined_spearman_correlations_f1'
    save(fig, output_dir / name)
    subset = results.loc[(results.method == method) & np.isclose(results.bond_fraction, 1)].copy()
    subset['abs_correlation'] = subset.correlation.abs()
    subset.to_csv(output_dir / (name + '.csv'), index=False)


def select_best_fold(cv_dir):
    """Select a compatible fold by held-out RMSE, breaking ties by model/fold."""
    configurations = json.loads((cv_dir / 'metadata.json').read_text())['configurations']
    names = [name for name, config in configurations.items()
             if config['columns'] == ['bond_fraction', 'O/N']]
    metrics = pd.read_csv(cv_dir / 'fold_metrics.csv')
    candidates = metrics.loc[metrics.model.isin(names) & np.isfinite(metrics.RMSE)]
    if candidates.empty:
        raise ValueError(f'No finite two-input fold RMSE in {cv_dir}')
    best = candidates.sort_values(['RMSE', 'model', 'fold']).iloc[0]
    path = cv_dir / best.model / f'fold_{int(best.fold):02d}.keras'
    selection = dict(model=best.model, fold=int(best.fold), RMSE=float(best.RMSE),
                     criterion='Lowest held-out fold RMSE among two-input models')
    return path, selection


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model_path, selection = select_best_fold(ROOT / 'neural_network_cross_validation')
    print(f'Model: {model_path}')
    model = tf.keras.models.load_model(model_path, compile=False)
    assert model.input_shape[-1] == 2
    fractions = np.round(np.arange(.9, 1.001, .02), 2)
    sources = sorted(ROOT.glob('NC_1D_*.csv'))
    if not sources:
        raise FileNotFoundError('No NC_1D_*.csv files found')
    metadata = dict(model=str(model_path), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    selection=selection,
                    architecture=[2, *[layer.units for layer in model.layers
                                        if isinstance(layer, tf.keras.layers.Dense)]], features=['bond_fraction', 'O/N'],
                    formulas={'O/N': 'O (%)/N (%)', 'DC1': '(4-2*(O/N))/(1+O/N), without clipping'},
                    bond_fractions=fractions.tolist(),
                    matching='Finite DC3, DC1, and measured outcome, independently per property; blank rows ignored.',
                    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    summaries, counts = [], []
    for source in sources:
        salt = source.stem.removeprefix('NC_1D_')
        raw = pd.read_csv(source)
        data = raw.dropna(how='all').copy()
        data.insert(0, 'source_row', data.index + 2)
        data.insert(0, 'source_file', source.name)
        oxygen = pd.to_numeric(data['O (%)'], errors='coerce')
        nitrogen = pd.to_numeric(data['N (%)'], errors='coerce')
        ratio = oxygen / nitrogen.replace(0, np.nan)
        valid = np.isfinite(ratio) & (oxygen >= 0) & (nitrogen > 0)
        data['model_O/N'] = ratio
        data['DC1'] = (4 - 2*ratio) / (1 + ratio)
        predictions = []
        for fraction in fractions:
            current = data.copy()
            current['bond_fraction'] = fraction
            current['predicted_largest_molecule_DC3'] = np.nan
            current['prediction_status'] = np.where(valid, 'ok', 'invalid_input')
            inputs = np.column_stack([np.full(valid.sum(), fraction), ratio.loc[valid]]).astype('float32')
            if len(inputs):
                prediction = model(inputs, training=False).numpy().ravel()
                assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
                current.loc[valid, 'predicted_largest_molecule_DC3'] = prediction
            predictions.append(current)
            for metric in ('R', 'F', 'B'):
                outcome = next(c for c in raw if c.startswith(metric + ' ('))
                pairs = current[['source_row', 'predicted_largest_molecule_DC3', 'DC1', outcome]].rename(
                    columns={'predicted_largest_molecule_DC3': 'DC3'}).apply(pd.to_numeric, errors='coerce')
                pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
                for method, function in [('pearson', pearsonr), ('spearman', spearmanr)]:
                    for descriptor in ('DC3', 'DC1'):
                        r, p = (function(pairs[descriptor], pairs[outcome]) if len(pairs) >= 3
                                and pairs[descriptor].nunique() > 1 and pairs[outcome].nunique() > 1 else (np.nan, np.nan))
                        if np.isfinite(r):
                            np.testing.assert_allclose(r, pairs[[descriptor, outcome]].corr(method=method).iloc[0, 1], atol=1e-10)
                        summaries.append(dict(dataset=salt, bond_fraction=fraction, metric=metric,
                                              descriptor=descriptor, method=method, n=len(pairs),
                                              correlation=r, p_value_two_sided=p))
                        if method == 'pearson' and descriptor == 'DC3' and np.isclose(fraction, 1):
                            scatter(pairs, salt, metric, outcome, r)
        pd.concat(predictions, ignore_index=True).to_csv(OUT / f'{source.stem}_DC3_predictions.csv', index=False)
        counts.append(dict(source=source.name, input_rows=len(raw), blank_rows=len(raw)-len(data),
                           valid_input_rows=int(valid.sum()), invalid_input_rows=int((~valid).sum())))
    metadata['row_counts'] = counts
    (OUT / 'prediction_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    results = pd.DataFrame(summaries)
    results.to_csv(OUT / 'DC3_DC1_correlations_all_f.csv', index=False)
    for method in ('pearson', 'spearman'):
        bars(results, method)
    print(pd.DataFrame(counts).to_string(index=False))
    print(results.loc[(results.method == 'pearson') & np.isclose(results.bond_fraction, 1)].to_string(index=False))
    print(f'Saved outputs to {OUT}')


if __name__ == '__main__':
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        main()
