#!/usr/bin/env python3
"""Fit largest-molecule DC3 versus system O/N and compare shared regressions."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit, differential_evolution


def fit1(x, a, b, c):
    return (1 - a*x) / (b + c*x)


def fit_fallback(x, A, B, pole):
    """Equivalent rational form for numerically unstable original fits."""
    return A + B / (x - pole)


def fit_record(x, y, parameters, covariance, warned=False):
    prediction = fit1(x, *parameters)
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite rational-fit predictions.')
    condition = np.linalg.cond(covariance) if np.isfinite(covariance).all() else np.inf
    a, b, c = parameters
    pole = -b/c if c != 0 else np.nan
    status = 'ill_conditioned' if warned or condition > 1e12 else 'ok'
    if x.min() <= pole <= x.max():
        status = 'pole_in_observed_range'
    return dict(zip(('a', 'b', 'c'), parameters)) | dict(
        zip(('a_SE', 'b_SE', 'c_SE'), np.sqrt(np.maximum(np.diag(covariance), 0)))) | dict(
        **scores(y, prediction), status=status, covariance_condition=condition, pole_O_N=pole)


def fallback_record(x, y, original):
    """Try poles on either side of the data and return a stable original-form fit.

    Convert both coefficients and local covariance to a,b,c so existing consumers
    retain the same equation and uncertainty interpretation.
    """
    span = np.ptp(x)
    gap = max(span * 1e-6, 1e-8)
    starts = []
    if all(key in original for key in ('a', 'b', 'c')) and original['c'] != 0:
        a, b, c = (original[key] for key in ('a', 'b', 'c'))
        starts.append(np.array([-a/c, (1+a*b/c)/c, -b/c]))
    for pole in [x.min()-span*k for k in (.1, .5, 2)] + [x.max()+span*k for k in (.1, .5, 2)]:
        A, B = np.linalg.lstsq(np.column_stack((np.ones(len(x)), 1/(x-pole))), y, rcond=None)[0]
        starts.append(np.array([A, B, pole]))
    candidates = []
    for initial in starts:
        if not np.isfinite(initial).all():
            continue
        if initial[2] < x.min()-gap:
            bounds = ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, x.min()-gap])
        elif initial[2] > x.max()+gap:
            bounds = ([-np.inf, -np.inf, x.max()+gap], [np.inf, np.inf, np.inf])
        else:
            continue
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', OptimizeWarning)
                q, covariance = curve_fit(fit_fallback, x, y, p0=initial,
                                          bounds=bounds, maxfev=3000)
            A, B, pole = q
            denominator = B-A*pole
            if abs(denominator) <= np.finfo(float).eps * max(1., abs(B), abs(A*pole)):
                continue
            numerator = np.array([-A, -pole, 1.])
            parameters = numerator / denominator
            jacobian = (np.array([[-1., 0., 0.], [0., 0., -1.], [0., 0., 0.]]) / denominator
                        - np.outer(numerator, [-pole, 1., -A]) / denominator**2)
            converted_covariance = jacobian @ covariance @ jacobian.T
            if not np.allclose(fit1(x, *parameters), fit_fallback(x, *q), rtol=1e-7, atol=1e-10):
                continue
            record = fit_record(x, y, parameters, converted_covariance, bool(caught))
            if record['status'] == 'ok':
                candidates.append(record)
        except (RuntimeError, ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue
    if not candidates:
        return None
    best = min(candidates, key=lambda record: record['RMSE'])
    if original['status'] == 'ill_conditioned' and best['RMSE'] > original['RMSE'] * (1+1e-6):
        return None
    return best


def scores(y, prediction):
    sse = np.sum((y-prediction)**2)
    sst = np.sum((y-np.mean(y))**2)
    return {'RMSE': float(np.sqrt(sse/len(y))),
            'MAE': float(np.mean(abs(y-prediction))),
            'R_squared': float(1-sse/sst) if sst > 0 else np.nan}


def prepare_data(source):
    df = pd.read_csv(source)
    df['O_N'] = ((5*df.T1_N + 4*df.T2_N + 3*df.T3_N) /
                 (2*df.M1_N + 2*df.M2_N))
    df['MPD_TMC'] = df.MPD/df.TMC
    finite = np.isfinite(df[['O_N', 'MPD_TMC', 'bond_fraction',
                             'largest_molecule_DC3']]).all(axis=1)
    df = df.loc[finite & (df.largest_molecule_DC3 > 0) & df.O_N.between(.75, 2)].copy()
    df['ratio_group'] = df.MPD_TMC.round(1)
    if df.empty:
        raise ValueError('No finite rows pass DC3 > 0 and 0.75 <= O/N <= 2.')
    return df.reset_index(drop=True)


def fit_groups(df):
    rows = []
    for ratio, group in df.groupby('ratio_group', sort=True):
        x = group.O_N.to_numpy()
        y = group.largest_molecule_DC3.to_numpy()
        row = dict(ratio_group=ratio, MPD_TMC_min=group.MPD_TMC.min(),
                   MPD_TMC_max=group.MPD_TMC.max(), n=len(group),
                   bond_fraction_min=group.bond_fraction.min(),
                   bond_fraction_max=group.bond_fraction.max(),
                   bond_fraction_mean=group.bond_fraction.mean(),
                   O_N_min=x.min(), O_N_max=x.max(), status='insufficient_data')
        if len(group) > 3 and len(np.unique(x)) >= 3:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', OptimizeWarning)
                    p, cov = curve_fit(fit1, x, y, maxfev=50000)
                row.update(fit_record(x, y, p, cov, bool(caught)))
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                row.update(status='fit_failed', error=str(exc))
            row.update(fit_method='original', original_status=row['status'])
            if row['status'] != 'ok':
                fallback = fallback_record(x, y, row)
                if fallback is not None:
                    if 'error' in row:
                        row['original_error'] = row.pop('error')
                    row.update(fallback, fit_method='rational_fallback')
                else:
                    row['fallback_status'] = 'no_stable_candidate'
        rows.append(row)
    result = pd.DataFrame(rows)
    for name in ('a', 'b', 'c'):
        if name not in result:
            result[name] = np.nan
    return result

def load_best_fold(cv_dir):
    """Select the two-input fold with lowest held-out RMSE (ties: model, fold)."""
    import tensorflow as tf

    metadata = json.loads((cv_dir/'metadata.json').read_text())
    names = [name for name, config in metadata['configurations'].items()
             if config['columns'] == ['bond_fraction', 'O/N']]
    metrics = pd.read_csv(cv_dir/'fold_metrics.csv')
    candidates = metrics.loc[metrics.model.isin(names) & np.isfinite(metrics.RMSE)]
    if candidates.empty:
        raise ValueError(f'No finite two-input fold RMSE in {cv_dir}')
    best = candidates.sort_values(['RMSE', 'model', 'fold']).iloc[0]
    path = cv_dir/best.model/f'fold_{int(best.fold):02d}.keras'
    model = tf.keras.models.load_model(path, compile=False)
    selection = dict(model=best.model, fold=int(best.fold), RMSE=float(best.RMSE),
                     model_path=str(path), columns=['bond_fraction', 'O/N'],
                     selection='Lowest held-out fold RMSE among two-input models',
                     caveat='Best-fold selection is optimistic; use all-fold metrics for evaluation.')
    return model, selection


def report_fixed_f_maxima(model, selection, output):
    """Search the plotted O/N domain, including both endpoints, at fixed f."""
    bounds = (.75, 2.)
    rows = []
    for fraction in (.8, .9, 1.):
        def objective(coords):
            x = np.asarray(coords).reshape(-1)
            features = np.column_stack((np.full_like(x, fraction), x)).astype('float32')
            prediction = model(features, training=False).numpy().ravel()
            if not np.isfinite(prediction).all():
                raise ValueError('Nonfinite DC3 predictions during maximum search.')
            return -prediction

        grid = np.linspace(*bounds, 10001)
        predictions = -objective(grid)
        index = int(np.argmax(predictions))
        candidates = [(float(predictions[index]), float(grid[index]))]
        for seed in (42, 123, 7):
            result = differential_evolution(
                objective, [bounds], seed=seed, popsize=35, maxiter=500,
                tol=1e-9, atol=1e-8, vectorized=True,
                updating='deferred', polish=False)
            candidates.append((-float(result.fun), float(result.x[0])))
        peak, position = max(candidates, key=lambda candidate: candidate[0])
        rows.append(dict(bond_fraction=fraction, O_N=position,
                         predicted_DC3=float(np.clip(peak, 0, 1)), raw_predicted_DC3=peak,
                         O_N_min=bounds[0], O_N_max=bounds[1],
                         model=selection['model'], fold=selection['fold'],
                         search='Three differential-evolution seeds plus 10001-point grid including endpoints'))
        print(f'Maximum predicted DC3 at f={fraction:.1f}: {peak:.8f} '
              f'(O/N={position:.8f}; 0.75 <= O/N <= 2)')
    maxima = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    maxima.to_csv(output/'DC3_ON_fixed_f_maxima.csv', index=False)
    return maxima


def plot_fixed_f_predictions(cv_dir, output, show=False):
    """Plot the best saved CV fold at fixed f, clipping displayed DC3 to [0, 1]."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.ticker import MultipleLocator

    model, selection = load_best_fold(cv_dir)
    print(f"Using {selection['model']} fold {selection['fold']} "
          f"(held-out RMSE={selection['RMSE']:.6g})")
    report_fixed_f_maxima(model, selection, output)
    fractions = (0.8, 0.9, 0.925, 0.95, 0.975, 1.0)
    x = np.linspace(.75, 2, 501)
    rows = []
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(3, 3))
        styles = ('-', '--', '-.', ':', (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)))
        for f, hue, style in zip(fractions, np.linspace(0, .75, len(fractions)), styles):
            features = np.column_stack((np.full_like(x, f), x)).astype('float32')
            # The saved network already includes its fold-specific normalization.
            raw = model(features, training=False).numpy().ravel()
            if not np.isfinite(raw).all():
                raise ValueError('Nonfinite DC3 predictions from selected fold.')
            y = np.clip(raw, 0, 1)
            ax.plot(x, y, color=hsv_to_rgb((hue, .8, .8)), linestyle=style,
                    linewidth=1.3, label=f'{f:g}')
            rows.append(pd.DataFrame({'bond_fraction': f, 'O_N': x, 'predicted_DC3': y,
                                      'raw_predicted_DC3': raw}))
        curves = pd.concat(rows, ignore_index=True)
        ax.set(xlabel='$r_{O/N}$', ylabel='Predicted DC$_3$', xlim=(.75, 2))
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MultipleLocator(.25))
        ax.yaxis.set_major_locator(MultipleLocator(.2))
        ax.legend(title='$f$', frameon=False, loc='upper right', fontsize=8,
                  title_fontsize=9, handlelength=2.4)
        fig.tight_layout(pad=.5)
        folder = output/'figures'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'DC3_ON_fixed_f.png'
        fig.savefig(path, dpi=300)
        curves.to_csv(output/'DC3_ON_fixed_f_predictions.csv', index=False)
        (output/'DC3_ON_prediction_model.json').write_text(
            json.dumps(selection, indent=2)+'\n')
        if show:
            plt.show()
        plt.close(fig)
    print(f'Saved fixed-f predictions: {path}')
    return curves

def make_figures(df, parameters,  output, figure, show,
                 prediction_curves=None, primary_only=False, prediction_label='Model prediction',
                 show_colorbar=True, prediction_grayscale=False,
                 x_limits=(.65, 2.1), x_ticks=(.8, 1.2, 1.6, 2.0), show_reference=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb, ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator
    r = df.ratio_group.to_numpy()
    lo, hi = r.min(), r.max()
    norm = Normalize(lo, hi if hi > lo else lo+1)
    cmap = ListedColormap(hsv_to_rgb(np.column_stack((np.linspace(0, .75, 256),
                                                    np.full(256, .8), np.full(256, .8)))))
    colors = cmap(norm(r))
    def scatter(ax, x, y):
        ax.scatter(x, y, s=15, facecolors='none', edgecolors=colors, linewidths=.6)
        ax.scatter(x, y, s=15, facecolors=colors, edgecolors='none', alpha=.2)
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(3, 3))
        scatter(ax, df.O_N, df.largest_molecule_DC3)
        for row in parameters.itertuples():
            if row.status not in ('ok', 'ill_conditioned'):
                continue
            x = np.linspace(row.O_N_min, row.O_N_max, 200)
            ax.plot(x, fit1(x, row.a, row.b, row.c), color='k', ls='--', lw=.65)
        if show_reference:
            x = np.linspace(1, 2, 100)
            ax.plot(x, (4-2*x)/(1+x), color=(.8, .4, .4), lw=1)
        if prediction_curves is not None:
            groups = list(prediction_curves.groupby('bond_fraction', sort=True))
            line_colors = [(0.075, 0.35, 0.75), (0.75, 0.32, 0.075)]
            line_styles = ['-', '-.']
            if prediction_grayscale:
                groups.reverse()
                line_colors = [str(v) for v in np.linspace(.2, .65, len(groups))]
                line_styles = [':'] * len(groups)
            for (fraction, curve), color, style in zip(
                    groups, line_colors, line_styles):
                ax.plot(curve['O/N'], curve.predicted_DC3, color=color,
                        ls=style, lw=1.2, label=f'$f={fraction:g}$', zorder=4)
            ax.legend(title=prediction_label, loc='upper right', frameon=False,
                      fontsize=7, title_fontsize=7, handlelength=1.7)
        ax.set(xlabel='$r_{O/N}$', ylabel='Largest-molecule DC$_3$',
               xlim=x_limits, ylim=(0, 1))
        ax.set_xticks(x_ticks)
        ax.set_yticks(np.arange(0, 1.01, .2))
        if show_colorbar:
            bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=.045, pad=.04)
            bar.set_label('MPD/TMC', fontsize=12)
        fig.tight_layout()
        figure.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure, dpi=300)
        if show:
            plt.show()
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('topology_results/systems.csv'))
    parser.add_argument('--output', type=Path, default=Path('topology_results'))
    parser.add_argument('--figure', type=Path, default=Path('Largest_DC3.png'))
    parser.add_argument('--no-show', action='store_true')
    parser.add_argument('--no-colorbar', action='store_true',
                        help='Hide the MPD/TMC colorbar on the main figure')
    parser.add_argument('--prediction-only', action='store_true',
                        help='Only plot the best CV fold at six fixed bond fractions')
    parser.add_argument('--cv-dir', type=Path,
                        default=Path('neural_network_cross_validation'),
                        help='Directory containing fold_metrics.csv, metadata.json and saved folds')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'topology-mpl'))
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    if args.prediction_only:
        plot_fixed_f_predictions(args.cv_dir, args.output, not args.no_show)
        return
    df = prepare_data(args.input)
    parameters = fit_groups(df)
    args.output.mkdir(parents=True, exist_ok=True)
    parameters.to_csv(args.output/'DC3_ON_fits_by_r.csv', index=False)
    linked = df[['system', 'MPD', 'TMC', 'MPD_TMC', 'ratio_group', 'bond_fraction',
                 'O_N', 'largest_molecule_DC3']].merge(
                     parameters[['ratio_group', 'a', 'b', 'c', 'status']],
                     on='ratio_group', how='left', validate='many_to_one')
    linked['group_fit_DC3'] = fit1(linked.O_N, linked.a, linked.b, linked.c)
    linked['group_fit_residual'] = linked.largest_molecule_DC3-linked.group_fit_DC3
    linked.to_csv(args.output/'DC3_ON_fit_parameters_by_system.csv', index=False)
    
    metadata = dict(input=str(args.input), n=len(df), ratio_groups=len(parameters),
                    target='largest_molecule_DC3', x='system O/N = (5*T1_N+4*T2_N+3*T3_N)/(2*M1_N+2*M2_N)',
                    r='exact initial MPD/TMC', f='bond_fraction',
                    p_T='f*min(2*r,3)/3', p_M='f*min(2*r,3)/(2*r)', s='p_T+p_M',
                    selection='finite x,r,f,target; target > 0; 0.75 <= x <= 2',
                    group_model='(1-a*x)/(b+c*x); groups round(r,1)',
                    group_fit_fallback='For failed, ill-conditioned, or in-range-pole fits, try A+B/(x-pole) '
                                       'with poles outside either end of the data; convert coefficients and covariance '
                                       'back to a,b,c. Keep stable candidates; ill-conditioned fits require no worse RMSE.',
                    uncertainty='SE from curve_fit local covariance; unreliable for ill_conditioned fits',
                    fitting='Unweighted least squares; each topology file has equal weight; no clipping',
                    validation='Leave one rounded MPD/TMC group out; pooled held-out metrics; per-fold scores in CSV',
                    caveats=['Positive-DC3 selection limits applicability; no validation on excluded systems.',
                             'Snapshots within a ratio are correlated; CV holds out entire ratios.',
                             'System O/N and initial composition differ from largest-molecule composition.',
                             'Functional-group conversion predictors are system-level empirical descriptors.',
                             'Model selection uses these CV results; no independent final test set.'],
    )

    prediction_curves = plot_fixed_f_predictions(args.cv_dir, args.output, show=False)
    overlay_curves = prediction_curves.loc[
        prediction_curves.bond_fraction.isin([1.0, 0.9, 0.8])]
    make_figures(df, parameters, args.output, args.figure, not args.no_show,
                 prediction_curves=overlay_curves.rename(columns={'O_N': 'O/N'}),
                 prediction_grayscale=True, prediction_label='NN prediction',
                 show_colorbar=not args.no_colorbar)
    print(f'Saved data and NN predictions: {args.figure}')

    (args.output/'DC3_ON_simplified_models.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'{len(df)} systems, {len(parameters)} ratio groups; saved parameters to {args.output}')
    print(parameters[['ratio_group', 'a', 'b', 'c', 'status']].to_string(index=False))


if __name__ == '__main__':
    main()
