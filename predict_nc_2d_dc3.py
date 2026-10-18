"""Predict NC_2D CSV rows with the best saved three-input CV fold.

Example: conda run -n tfml python predict_nc_2d_dc3.py --bond-fractions 1.0 0.9 0.8
Defaults to f=1.0 and the fold with the lowest held-out RMSE.
Inputs use N2/(N2+O1) for O2/O and (O1+N2)/(N2+N1+N3) for O/N.
Original files are preserved; outputs contain one row per input row and f.
Also saves DC3/DC1/HABD Pearson and Spearman correlations for NaCl R, F, and B,
with correlation bars and property scatter plots at f=1.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dc3_mpl'))
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import pearsonr, spearmanr
from predict_nc_1d_dc3 import bars, scatter, plt


def correlate_predictions(predictions, output):
    """Compare descriptors on shared finite rows for each NaCl property."""
    summaries = []
    habd = next(c for c in predictions if c.startswith('HABD'))
    for fraction, current in predictions.groupby('bond_fraction', sort=True):
        for metric in ('R', 'F', 'B'):
            outcome = next(c for c in current if c.startswith(metric + ' ('))
            pairs = current[['source_row', 'predicted_largest_molecule_DC3', 'DC1', habd, outcome]].rename(
                columns={'predicted_largest_molecule_DC3': 'DC3', habd: 'HABD'}).apply(pd.to_numeric, errors='coerce')
            pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna()
            for method, function in [('pearson', pearsonr), ('spearman', spearmanr)]:
                for descriptor in ('DC3', 'DC1', 'HABD'):
                    r, p = (function(pairs[descriptor], pairs[outcome]) if len(pairs) >= 3
                            and pairs[descriptor].nunique() > 1 and pairs[outcome].nunique() > 1
                            else (np.nan, np.nan))
                    if np.isfinite(r):
                        np.testing.assert_allclose(
                            r, pairs[[descriptor, outcome]].corr(method=method).iloc[0, 1], atol=1e-10)
                    summaries.append(dict(dataset='NaCl', bond_fraction=fraction, metric=metric,
                                          descriptor=descriptor, method=method, n=len(pairs),
                                          correlation=r, p_value_two_sided=p))
                    if method == 'pearson' and descriptor == 'DC3' and np.isclose(fraction, 1):
                        scatter(pairs, 'NaCl', metric, outcome, r, output_dir=output)
    results = pd.DataFrame(summaries)
    results.to_csv(output / 'DC3_DC1_correlations_all_f.csv', index=False)
    for method in ('pearson', 'spearman'):
        bars(results, method, output_dir=output, descriptors=('DC3', 'DC1', 'HABD'))


def select_best_fold(cv_dir):
    """Select a compatible fold by held-out RMSE, breaking ties by model/fold."""
    configurations = json.loads((cv_dir / 'metadata.json').read_text())['configurations']
    names = [name for name, config in configurations.items()
             if config['columns'] == ['bond_fraction', 'O/N', 'O2/O']]
    metrics = pd.read_csv(cv_dir / 'fold_metrics.csv')
    candidates = metrics.loc[metrics.model.isin(names) & np.isfinite(metrics.RMSE)]
    if candidates.empty:
        raise ValueError(f'No finite three-input fold RMSE in {cv_dir}')
    best = candidates.sort_values(['RMSE', 'model', 'fold']).iloc[0]
    path = cv_dir / best.model / f'fold_{int(best.fold):02d}.keras'
    selection = dict(model=best.model, fold=int(best.fold), RMSE=float(best.RMSE),
                     criterion='Lowest held-out fold RMSE among three-input models')
    return path, selection


def make_nc_features(data, fraction):
    n2 = pd.to_numeric(data['N2 (%)'], errors='coerce')
    o1 = pd.to_numeric(data['O1 (%)'], errors='coerce')
    n13 = pd.to_numeric(data['N1+N3 (%)'], errors='coerce')
    oxygen, nitrogen = n2 + o1, n2 + n13
    features = pd.DataFrame({
        'bond_fraction': fraction,
        'O/N': oxygen / nitrogen.replace(0, np.nan),
        'O2/O': n2 / oxygen.replace(0, np.nan),
    })
    valid = (np.isfinite(features).all(axis=1) & (oxygen > 0) & (nitrogen > 0)
             & (n2 >= 0) & (o1 >= 0) & (n13 >= 0))
    return features, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bond-fractions', '--bond-fraction', type=float, nargs='+',
                        default=[1.0], help='Bond fractions to predict (default: 1.0).')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if not all(np.isfinite(f) and 0 <= f <= 1 for f in args.bond_fractions):
        parser.error('Bond fractions must be finite and between 0 and 1.')
    root = Path(__file__).resolve().parent
    sources = sorted(root.glob('NC_2D_*.csv'))
    if not sources:
        raise FileNotFoundError('No NC_2D_*.csv files found.')
    output = args.output_dir or root / 'neural_network_cross_validation' / 'nc_2d_predictions'
    output.mkdir(parents=True, exist_ok=True)
    if args.model is None:
        model_path, selection = select_best_fold(root / 'neural_network_cross_validation')
    else:
        model_path, selection = args.model, dict(criterion='Explicit --model override')
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f'Model: {model_path}')
    metadata = dict(model=str(model_path.resolve()), model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    selection=selection,
                    bond_fractions=args.bond_fractions, features=['bond_fraction', 'O/N', 'O2/O'],
                    formulas={'O/N':'(O1+N2)/(N2+N1+N3)', 'O2/O':'N2/(N2+O1)',
                              'DC1':'(4-2*(O (%)/N (%)))/(1+O (%)/N (%)), without clipping'},
                    correlation_matching='Finite DC3, DC1, HABD, and measured outcome, independently per NaCl property.',
                    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output / 'prediction_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for source in sources:
        data = pd.read_csv(source)
        data = data.dropna(how='all')
        rows = []
        for fraction in args.bond_fractions:
            features, valid = make_nc_features(data, fraction)
            result = data.copy()
            result.insert(0, 'source_row', data.index.to_numpy() + 2)
            result.insert(0, 'source_file', source.name)
            result['bond_fraction'] = fraction
            result['model_O/N'] = features['O/N']
            result['model_O2/O'] = features['O2/O']
            elemental_ratio = (pd.to_numeric(data['O (%)'], errors='coerce') /
                               pd.to_numeric(data['N (%)'], errors='coerce').replace(0, np.nan))
            result['DC1'] = (4 - 2 * elemental_ratio) / (1 + elemental_ratio)
            result['predicted_largest_molecule_DC3'] = np.nan
            result['prediction_status'] = np.where(valid, 'ok', 'invalid_input')
            if valid.any():
                prediction = model(features.loc[valid].to_numpy(dtype='float32'),
                                   training=False).numpy().ravel()
                assert np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all()
                result.loc[valid, 'predicted_largest_molecule_DC3'] = prediction
            rows.append(result)
        destination = output / f'{source.stem}_DC3_predictions.csv'
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(destination, index=False)
        if source.stem == 'NC_2D_NaCl':
            correlate_predictions(combined, output)
        print(f'{source.name}: {len(data)} source rows, {len(combined)} output rows, '
              f'{(combined.prediction_status == "ok").sum()} valid predictions -> {destination.name}')


if __name__ == '__main__':
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 12, 'axes.spines.top': False,
                         'axes.spines.right': False}):
        main()
