"""3D simulation DC3 scatter and best-CV-fold neural-network surface at f=1.

Run: conda run -n tfml python plot_dc3_3d_surface.py
Surface is restricted by the three user-supplied composition inequalities.
The 2D model uses its lowest-test-RMSE fold, as in plot_dc3_parity.py.
This best-fold illustration is optimistic; use all-fold metrics for evaluation.
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
from matplotlib.colors import hsv_to_rgb, Normalize, LightSource
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import tensorflow as tf


def constraint_values(x, y):
    # x = O/N ratio, y = O2/O ratio

    denominator = 3 + x*(y + 1)
    a = 6*(1 - x*y)/denominator # n_M1
    b = 3*(2*x*y - 1)/denominator #n_M2
    # n_T2 = - 2*n_T1 + A
    # 0 < n_T2 --> (i) n_T1 < A/2
    # n_T2 < 1 -->  (ii) n_T1 > (A-1)/2

    # n_T3 = n_T1 + B
    # n_T3 < 1 -> (iii) n_T1 < 1-B
    # 0 < n_T3 -> (iv)  n_T1 > -B

    # max(-B, (A-1)/2) < n_T1 < min( A/2, 1-B) 
    A= 3*x*(1-y)/denominator
    B= 2*x*(2*y-1)/denominator
    c_left = np.maximum(-B,(A-1)/2)
    c_right = np.minimum( A/2, 1-B)

    return a, b, c_left, c_right


def allowed(x, y):
    a, b, left, right = constraint_values(x, y)
    return (a > 0) & (a < 1) & (b > 0) & (b < 1) & (left < right)


def load_best_2d_model(folder):
    """Load the saved three-input architecture's lowest-test-RMSE fold."""
    metrics = pd.read_csv(folder / 'fold_metrics.csv')
    names = [name for name in metrics.model.dropna().unique()
             if name.startswith('three_inputs_')]
    if len(names) != 1:
        raise ValueError(f'Expected one three-input CV architecture; found {names}')
    candidates = metrics.loc[(metrics.model == names[0]) & np.isfinite(metrics.RMSE)]
    if candidates.empty:
        raise ValueError(f'No finite fold RMSE values for {names[0]}')
    best = candidates.sort_values(['RMSE', 'fold']).iloc[0]
    model_path = folder / names[0] / f'fold_{int(best.fold):02d}.keras'
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.input_shape[-1] != 3:
        raise ValueError(f'Expected inputs [bond_fraction, O/N, O2/O]: {model_path}')
    selection = dict(model=names[0], fold=int(best.fold),
                     criterion='lowest test RMSE', RMSE=float(best.RMSE))
    return model, model_path, selection


def main():
    root = Path(__file__).resolve().parent
    cv_folder = root / 'neural_network_cross_validation'
    model, model_path, selection = load_best_2d_model(cv_folder)
    print(f'Selected CV model: {model_path} (test RMSE={selection["RMSE"]:.8f})')
    source = root / 'topology_results/systems.csv'
    out = cv_folder / 'three_dimensional_surface'
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    oxygen = 5*data.T1_N + 4*data.T2_N + 3*data.T3_N
    data['O/N'] = oxygen / (2*data.M1_N + 2*data.M2_N).replace(0, np.nan)
    data['O2/O'] = (data.T1_N + 2*data.T2_N + 3*data.T3_N) / oxygen.replace(0, np.nan)
    data['MPD/TMC'] = data.MPD / data.TMC
    valid = np.isfinite(data[['O/N', 'O2/O', 'bond_fraction', 'largest_molecule_DC3', 'MPD/TMC']]).all(axis=1)
    data = data.loc[valid & (data.largest_molecule_monomers >= 100)
                    & (data.largest_molecule_DC3 >= 0) & data['O/N'].between(.65, 2)].copy()
    data.to_csv(out / 'scatter_data.csv', index=False)
    x, y = np.meshgrid(np.linspace(.75, 2, 151), np.linspace(.5, 1, 151))
    xy = np.column_stack([x.ravel(), y.ravel()])
    valid_surface = allowed(xy[:, 0], xy[:, 1])
    if not valid_surface.any():
        raise ValueError('No feasible surface points at f=1; check composition constraints.')
    inputs = np.column_stack([np.ones(valid_surface.sum()), xy[valid_surface]]).astype('float32')
    z = np.full(len(xy), np.nan)
    z[valid_surface] = model(inputs, training=False).numpy().ravel()
    assert np.isfinite(z[valid_surface]).all() and ((z[valid_surface] >= 0) & (z[valid_surface] <= 1)).all()
    pd.DataFrame({'bond_fraction': 1., 'O/N': xy[valid_surface, 0], 'O2/O': xy[valid_surface, 1],
                  'predicted_DC3': z[valid_surface]}).to_csv(out / 'surface_f1.csv', index=False)

    def objective(coords):
        pts = np.asarray(coords).T
        keep = allowed(pts[:, 0], pts[:, 1])
        result = np.full(len(pts), 10.)
        if keep.any():
            features = np.column_stack([np.ones(keep.sum()), pts[keep]]).astype('float32')
            result[keep] = -model(features, training=False).numpy().ravel()
        return result

    # Independent global searches plus a dense grid check for narrow maxima.
    candidates = []
    for seed in (42, 123, 7):
        opt = differential_evolution(objective, [(.75, 2), (.5, 1)], seed=seed,
                                     popsize=35, maxiter=500, tol=1e-9, atol=1e-8,
                                     vectorized=True, updating='deferred', polish=False)
        candidates.append((-float(opt.fun), *map(float, opt.x)))
    gx, gy = np.meshgrid(np.linspace(.75, 2, 1001), np.linspace(.5, 1, 1001))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    scores = -objective(grid.T)
    best_grid = int(np.argmax(scores))
    candidates.append((float(scores[best_grid]), *map(float, grid[best_grid])))
    peak, peak_x, peak_y = max(candidates)
    assert allowed(peak_x, peak_y)
    a, b, left, right = constraint_values(peak_x, peak_y)
    maximum = dict(bond_fraction=1., predicted_DC3=peak, O_N=peak_x, O2_O=peak_y,
                   constraint_a=a, constraint_b=b, constraint_c_left=left, constraint_c_right=right,
                   search='Three differential-evolution seeds plus 1001 x 1001 grid; strict inequalities')
    (out / 'constrained_maximum_f1.json').write_text(json.dumps(maximum, indent=2) + '\n')
    print(f'Maximum predicted DC3 at f=1: {peak:.8f} '
          f'(O/N={peak_x:.8f}, O2/O={peak_y:.8f})')
    print(json.dumps(maximum, indent=2))
    ratio = data['MPD/TMC'].round(1)
    hue = .75 * Normalize(ratio.min(), ratio.max())(ratio)
    colors = hsv_to_rgb(np.column_stack([hue, np.full(len(data), .8), np.full(len(data), .8)]))
    with plt.rc_context({'font.size': 12, 'axes.labelsize': 14}):
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('persp', focal_length=.9)
        ax.plot_surface(x, y, z.reshape(x.shape), color='.55', alpha=.20,
                        rstride=6, cstride=6, linewidth=.35, edgecolor='.3',
                        antialiased=True, shade=True,
                        lightsource=LightSource(azdeg=315, altdeg=40))
        ax.scatter(data['O/N'], data['O2/O'], data.largest_molecule_DC3,
                   s=13, facecolors='none', edgecolors=colors, linewidths=.5,
                   depthshade=True)
        ax.scatter(data['O/N'], data['O2/O'], data.largest_molecule_DC3,
                   s=13, c=colors, edgecolors='none', alpha=.80,
                   depthshade=True)
        ax.set(xlabel=r'$r_{O/N}$', ylabel=r'$r_{O_2/O}$',
               zlabel=r'Largest-molecule $DC_3$', xlim=(.65, 2), zlim=(0, 1))
        ax.set_xticks([.8, 1.2, 1.6, 2.])
        ax.set_zticks(np.arange(0, 1.01, .2))
        ax.set_ylim(.4, 1.)
        ax.set_yticks([.5, .7, .9])
        ax.view_init(elev=24, azim=-8)
        ax.set_box_aspect((1.1, 1, .95), zoom=.86)
        ax.grid(True)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((.96, .96, .96, .35))
            axis._axinfo['grid'].update(color=(.65, .65, .65, .5), linewidth=.5,
                                        linestyle='-')
            axis.labelpad = 9
        ax.legend(handles=[Line2D([], [], marker='o', linestyle='none', color='.3',
                                  markerfacecolor='none', markersize=4, label='Simulation'),
                           Patch(facecolor='.55', edgecolor='.3', alpha=.20,
                                 label=r'Neural network ($f=1$)')],
                  loc='upper right', frameon=False, fontsize=11)
        fig.subplots_adjust(left=0, right=.91, bottom=.05, top=1)
        for ext in ['png']:
            fig.savefig(root / f'Largest_DC3_neural_network_3D.{ext}', dpi=300)
        plt.close(fig)
    architecture = [3] + [layer.units for layer in model.layers
                          if isinstance(layer, tf.keras.layers.Dense)]
    metadata = dict(model=str(model_path), architecture=architecture,
                    cross_validation_selection=selection,
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    features=['bond_fraction', 'O/N', 'O2/O'], surface_bond_fraction=1.,
                    surface_alpha=.20, scatter_fill_alpha=.80,
                    scatter_rows=len(data), surface_points=int(valid_surface.sum()),
                    scatter_filter='finite; monomers >= 100; DC3 >= 0; 0.65 <= O/N <= 2',
                    surface_domain='0.75 <= O/N <= 2; 0.5 <= O2/O <= 1; f=1; all three strict composition inequalities.',
                    constraints=['0 < 6*(1-x*y)/(3+x*(y+1)) < 1',
                                 '0 < 3*(2*x*y-1)/(3+x*(y+1)) < 1',
                                 'max(-B, (A-1)/2) < min(A/2, 1-B); '
                                 'A=3*x*(1-y)/(3+x*(y+1)); B=2*x*(2*y-1)/(3+x*(y+1))'],
                    maximum=maximum,
                    scatter_color='Rounded MPD/TMC, HSV hue 0 to 0.75, consistent with 2D reference')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved 3D figure: {len(data)} scatter rows, {valid_surface.sum()} surface predictions')
    print(data[['O/N', 'O2/O', 'bond_fraction']].agg(['min', 'max']).to_string())


if __name__ == '__main__':
    main()
