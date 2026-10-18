"""Overlay pooled 1D and 2D cross-validation-model predictions at f=1.

Run: conda run -n tfml python plot_nc_dc3_histogram_f1.py
Each distribution is normalized to unit area (probability density).
Includes all valid source rows in each dataset family, without deduplication.
"""
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dc3_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parent
    prediction_root = root / 'neural_network_cross_validation'
    output = prediction_root / 'combined_predictions/histograms'
    output.mkdir(parents=True, exist_ok=True)
    bins = np.arange(-14, 18) * .1
    counts, points = [], []
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(3, 3), layout='constrained')
        for name, folder, color, style in [
                ('1D', 'nc_1d_predictions', hsv_to_rgb((.58, .9, .75)), '-'),
                ('2D', 'nc_2d_predictions', hsv_to_rgb((1/3, 1., .75)), '-')]:
            sources = sorted((prediction_root / folder).glob(f'NC_{name}_*_DC3_predictions.csv'))
            if not sources:
                raise FileNotFoundError(f'No {name} predictions')
            pooled = pd.concat([pd.read_csv(p) for p in sources], ignore_index=True)
            selected = pooled.loc[np.isclose(pooled.bond_fraction, 1)
                                  & pooled.prediction_status.eq('ok')].copy()
            if selected.empty:
                raise ValueError(f'No valid {name} predictions at f=1')
            values = selected.predicted_largest_molecule_DC3.to_numpy()
            assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
            assert not selected.duplicated(['source_file', 'source_row']).any()
            heights, _ = np.histogram(values, bins)
            assert heights.sum() == len(values)
            density = heights / (len(values) * np.diff(bins))
            np.testing.assert_allclose(np.sum(density * np.diff(bins)), 1.)
            ax.stairs(density, bins, fill=True, color=color, alpha=.22, linewidth=0)
            ax.stairs(density, bins, color=color, linewidth=1.2, linestyle=style,
                      label=f'{name} $DC_3$ ($n={len(values)}$)')
            counts.extend(dict(model=name, descriptor='DC3', bond_fraction=1., bin_left=left, bin_right=right, count=int(n), density=float(d))
                          for left, right, n, d in zip(bins[:-1], bins[1:], heights, density))
            selected['DC1'] = pd.to_numeric(selected['DC1'], errors='raise')
            dc1_counts, _ = np.histogram(selected.DC1, bins)
            assert np.isfinite(selected.DC1).all() and dc1_counts.sum() == len(selected)
            dc1_density = dc1_counts / (len(selected) * np.diff(bins))
            np.testing.assert_allclose(np.sum(dc1_density * np.diff(bins)), 1.)
            ax.stairs(dc1_density, bins, fill=False, color=color, linewidth=1.2,
                      linestyle=':', label=f'{name} $DC_1$ ($n={len(values)}$)', zorder=4)
            counts.extend(dict(model=name, descriptor='DC1', bond_fraction=1., bin_left=left, bin_right=right, count=int(n), density=float(d))
                          for left, right, n, d in zip(bins[:-1], bins[1:], dc1_counts, dc1_density))
            selected['model'] = name
            points.append(selected[['model', 'source_file', 'source_row', 'bond_fraction',
                                    'predicted_largest_molecule_DC3', 'DC1']])
            print(f'{name}: n={len(values)}, sources={[p.name for p in sources]}')
        ax.set(xlim=(bins[0], bins[-1]), ylim=(0, None), xlabel=r'DC', ylabel='Probability density')
        ax.set_xticks(np.arange(-1, 1.51, .5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(y=.25)
        #ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02),
        #          ncol=2, frameon=False, fontsize=7, columnspacing=1, handlelength=1.8)
        ax.set_xlim(-0.9,1.6)
        stem = output / 'NC_1D_2D_DC3_histogram_f1'
        for ext in ['png']:
            fig.savefig(stem.with_suffix('.' + ext), dpi=300)
        plt.close(fig)
    pd.DataFrame(counts).to_csv(stem.with_suffix('.csv'), index=False)
    pd.concat(points, ignore_index=True).to_csv(output / 'NC_1D_2D_DC3_histogram_f1_predictions.csv', index=False)


if __name__ == '__main__':
    main()
