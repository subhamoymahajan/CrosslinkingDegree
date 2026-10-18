"""Plot DC3_0 from NC_2D*.csv with resolved species.

Run: python3 plot_dc3_0_distribution.py
Requires N2 (%), O1 (%), and N1+N3 (%) (or separate N1/N3).
Files without these species are reported, never assigned assumed values.
Histogram areas are one per dataset family; rows have equal weight.
"""
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'dc3_0_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def load_data(root):
    frames, reports = [], []
    for family in ('2D',):
        for source in sorted(root.glob(f'NC_{family}*.csv')):
            raw = pd.read_csv(source)
            data = raw.dropna(how='all').copy()
            report = dict(source=source.name, rows=len(data), blank_rows=len(raw)-len(data))
            required = ['N2 (%)', 'O1 (%)']
            required += (['N1+N3 (%)'] if 'N1+N3 (%)' in data
                         else ['N1 (%)', 'N3 (%)'])
            missing = [c for c in required if c not in data]
            if missing:
                reports.append(dict(report, status='missing_species', missing_columns=missing))
                continue
            numeric = data[required].apply(pd.to_numeric, errors='coerce')
            n2, o1 = numeric['N2 (%)'], numeric['O1 (%)']
            n13 = (numeric['N1+N3 (%)'] if 'N1+N3 (%)' in numeric
                   else numeric['N1 (%)'] + numeric['N3 (%)'])
            oxygen, nitrogen = n2 + o1, n2 + n13
            valid = (np.isfinite(numeric).all(axis=1) & (numeric >= 0).all(axis=1)
                     & (oxygen > 0) & (nitrogen > 0))
            ron = oxygen / nitrogen.replace(0, np.nan)
            ro2 = n2 / oxygen.replace(0, np.nan)
            dc = (10*ron*ro2 - 2*ron - 6) / (ron*(ro2+1))
            # Independent algebraic check of the requested ratio expression.
            simplified = (2*n2 - 2*o1 - 6*n13) / (2*n2 + o1)
            np.testing.assert_allclose(dc[valid], simplified[valid], atol=1e-12)
            data.insert(0, 'source_row', data.index+2)
            data.insert(0, 'source_file', source.name)
            data.insert(0, 'family', family)
            data['rON'], data['rO2'] = ron, ro2
            data['DC3_0'] = dc.where(valid)
            data['status'] = np.where(valid, 'ok', 'invalid_species')
            frames.append(data)
            reports.append(dict(report, status='calculated', valid_rows=int(valid.sum()),
                                invalid_rows=int((~valid).sum())))
    if not frames:
        raise ValueError(f'No species-resolved datasets: {reports}')
    return pd.concat(frames, ignore_index=True), reports


def main():
    root = Path(__file__).resolve().parent
    output = root / 'dc3_0_distribution'
    output.mkdir(exist_ok=True)
    data, reports = load_data(root)
    valid = data.loc[data.status.eq('ok')]
    values = valid.DC3_0.to_numpy()
    if not len(values):
        raise ValueError('No valid DC3_0 values')
    # Shared 0.1-wide bins; retain the full range without clipping to [0, 1].
    low = np.floor(values.min()*10)/10
    high = np.floor(values.max()*10)/10 + .1
    bins = np.linspace(low, high, int(round((high-low)/.1))+1)
    histograms = []
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42}):
        fig, ax = plt.subplots(figsize=(3, 3), layout='constrained')
        for family, group in valid.groupby('family'):
            counts, _ = np.histogram(group.DC3_0, bins=bins)
            density = counts / (len(group)*np.diff(bins))
            assert counts.sum() == len(group)
            np.testing.assert_allclose(np.sum(density*np.diff(bins)), 1.)
            color = hsv_to_rgb((.58 if family == '1D' else 1/3, .9, .7))
            ax.stairs(density, bins, fill=True, color=color, alpha=.2, linewidth=0)
            ax.stairs(density, bins, color=color, linewidth=1.3,
                      linestyle='-' if family == '1D' else '--',
                      label=f'{family} ($n={len(group)}$)')
            histograms.extend(dict(family=family, bin_left=l, bin_right=r,
                                   count=int(c), probability=c/len(group), density=d)
                              for l, r, c, d in zip(bins[:-1], bins[1:], counts, density))
        ax.set(xlabel=r'$DC_{3,0}$', ylabel='Probability density', ylim=(0, None))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(x=.03, y=.25)
        ax.legend(loc='upper right', frameon=False, fontsize=8)
        for ext in ('png', 'svg'):
            fig.savefig(output / f'NC_DC3_0_distribution.{ext}', dpi=300)
        plt.close(fig)
    data.to_csv(output / 'DC3_0_values.csv', index=False)
    pd.DataFrame(histograms).to_csv(output / 'DC3_0_histogram.csv', index=False)
    metadata = dict(formula='(10*rON*rO2 - 2*rON - 6)/(rON*(rO2+1))',
                    rON='(N2+O1)/(N2+N1+N3)', rO2='N2/(N2+O1)',
                    bin_width=.1, normalization='Unit area per family; equal weight per row',
                    clipping=False, sources=reports)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(reports, indent=2))
    print(valid.groupby('family').DC3_0.agg(['count', 'min', 'max', 'mean']).to_string())
    print(f'Saved to {output}')


if __name__ == '__main__':
    main()
