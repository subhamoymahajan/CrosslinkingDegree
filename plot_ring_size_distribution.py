#!/usr/bin/env python3
"""Plot exact cycle-size counts normalized by (TMC1+TMC2+TMC3)/2."""
import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from find_residue_cycles import analyze
from analyze_topology import read_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topology', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path('topology_results/cycles')/args.topology.parent.name/args.topology.stem
    report = analyze(args.topology, output)
    sizes = np.array([cycle['size'] for cycle in report['cycles']], dtype=int)
    if not len(sizes):
        parser.exit(0, 'No basis cycles; no ring-size distribution to plot.\n')
    graph = read_graph(args.topology)
    tmc_counts = {f'TMC{state}': sum(attrs['species'] == 'TMC' and attrs['state'] == state
                                   for _, attrs in graph.nodes(data=True))
                  for state in (1, 2, 3)}
    maximum_rings = sum(tmc_counts.values()) / 2
    if maximum_rings <= 0:
        parser.error('Maximum ring count must be positive for normalization.')
    exact = Counter(sizes.tolist())
    # MPD/TMC connectivity is bipartite, so only even ring sizes are possible.
    # Include unobserved even sizes explicitly as zero counts, without binning.
    ring_sizes = np.arange(min(4, sizes.min()), max(6, sizes.max())+1, 2)
    counts = np.array([exact.get(int(size), 0) for size in ring_sizes])
    assert counts.sum() == len(sizes) == report['cycle_rank']
    normalized_counts = counts / maximum_rings
    with (output/'ring_size_distribution.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['ring_size_residues', 'cycle_count', 'fraction', 'maximum_rings', 'normalized_count'])
        for size, count in zip(ring_sizes, counts):
            writer.writerow([int(size), int(count), count/len(sizes), maximum_rings, count/maximum_rings])
    summary = dict(topology=report['topology'], method=report['method'],
                   cycle_count=len(sizes), min_size=int(sizes.min()), max_size=int(sizes.max()),
                   median_size=float(np.median(sizes)), mean_size=float(sizes.mean()),
                   size_unit='residues', plot='Exact counts at each even size; absent sizes have zero count; logarithmic x-axis',
                   TMC_counts=tmc_counts, maximum_rings=maximum_rings,
                   normalization='cycle_count / ((TMC1+TMC2+TMC3)/2); TMC0 excluded',
                   total_normalized_count=float(normalized_counts.sum()),
                   highlighted_size=6, highlighted_count=exact.get(6, 0),
                   highlighted_normalized_count=exact.get(6, 0)/maximum_rings,
                   note='Fundamental cycle basis; not all rings or a minimum cycle basis.')
    (output/'ring_size_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'topology-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
    with plt.rc_context({'font.size': 9, 'axes.labelsize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(ring_sizes, normalized_counts, color=hsv_to_rgb((.6, .8, .8)), linewidth=.9)
        ax.axvline(6, color='0.35', linestyle=':', linewidth=1.2)
        ax.axvline(12, color='0.35', linestyle=':', linewidth=1.2)
        ax.text(6, .98, '6', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=9, backgroundcolor='white')
        ax.text(12, .98, '12', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=9, backgroundcolor='white')
        ax.set_xscale('log', base=2)
        ax.set(xlim=(ring_sizes[0], 2**np.ceil(np.log2(ring_sizes[-1]))),
               ylim=(0, max(normalized_counts)*1.18),
               xlabel='Ring size (n)', ylabel=r'$R(n)/R_{\mathrm{max}}$')
        ax.set_xticks(2.0**np.arange(2, int(np.ceil(np.log2(ring_sizes[-1])))+1, 2))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.text(.97, .96, f'$N={len(sizes)}$', transform=ax.transAxes, ha='right', va='top')
        fig.tight_layout(pad=.5)
        fig.savefig(output/'ring_size_distribution.png', dpi=300)
        plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f'Saved {output}/ring_size_distribution.png')


if __name__ == '__main__':
    main()
