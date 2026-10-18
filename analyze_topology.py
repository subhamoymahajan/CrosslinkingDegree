#!/usr/bin/env python3
"""Analyze explicit, single-moleculetype MPD/TMC topologies; see TOPOLOGY_USAGE.md."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from itertools import combinations
import json
import math
import os
from pathlib import Path
import re
import tempfile

import networkx as nx

STATE = re.compile(r'^(MPD|TMC)([0-3])?$')
COUNTS = ('M1', 'M2', 'T1', 'T2', 'T3')
METRICS = ('th', 'n', 'e', 'DC3')
RATIOS = tuple(f'{key}_N' for key in COUNTS)


class UnsupportedTopology(ValueError):
    """Input needs external preprocessing or system expansion."""


def divide(a, b):
    return a / b if b else math.nan


def measures(counts):
    m1, m2, t1, t2, t3 = (counts[k] for k in COUNTS)
    total = sum(counts[key] for key in COUNTS)
    return dict(th=divide(t1, t1 + m1), n=divide(t3, t2 + t3),
                e=divide(t3 - t1 - m1, t3), DC3=divide(t3 - t1 - m1, t2 + t3),
                **{f'{key}_N': divide(counts[key], total) for key in COUNTS})


def read_graph(path):
    """Keep residue IDs/names and deduplicate atomic and residue edges.

    Multiple atomic links between a residue pair are counted for functionality,
    while the residue connectivity graph has one edge per pair.
    """
    graph = nx.Graph()
    atoms, atomic_edges = {}, set()
    section = ''
    atom_sections = 0
    molecule_types = 0
    with path.open() as source:
        for number, line in enumerate(source, 1):
            stripped = line.lstrip()
            if stripped.startswith('#'):
                raise UnsupportedTopology(f'{path}:{number}: preprocess topology directives first')
            if stripped.startswith('['):
                section = stripped.split(']', 1)[0][1:].strip()
                atom_sections += section == 'atoms'
                molecule_types += section == 'moleculetype'
                if atom_sections > 1 or molecule_types > 1:
                    raise UnsupportedTopology(f'{path}: multiple molecule types are unsupported')
                continue
            if section not in ('atoms', 'bonds', 'molecules'):
                continue
            fields = line.split(';', 1)[0].split()
            if not fields:
                continue
            try:
                if section == 'atoms':
                    atom, residue, name = int(fields[0]), int(fields[2]), fields[3]
                    if atom in atoms:
                        raise ValueError(f'duplicate atom ID {atom}')
                    atoms[atom] = residue
                    if residue in graph and graph.nodes[residue]['residue_name'] != name:
                        raise ValueError(f'conflicting names for residue {residue}')
                    match = STATE.fullmatch(name)
                    if match:
                        species, state = match.groups()
                        if species == 'MPD' and state == '3':
                            raise ValueError('MPD3 is not an allowed state')
                        graph.add_node(residue, residue_name=name, species=species,
                                       declared_state=int(state) if state else None)
                elif section == 'bonds':
                    a, b = map(int, fields[:2])
                    ra, rb = atoms[a], atoms[b]
                    if ra == rb or ra not in graph or rb not in graph:
                        continue
                    edge = tuple(sorted((a, b)))
                    if edge in atomic_edges:
                        continue
                    atomic_edges.add(edge)
                    if graph.nodes[ra]['species'] == graph.nodes[rb]['species']:
                        raise ValueError('inter-residue bond joins identical species')
                    if graph.has_edge(ra, rb):
                        graph[ra][rb]['atomic_bonds'] += 1
                    else:
                        graph.add_edge(ra, rb, atomic_bonds=1)
                elif int(fields[1]) != 1:
                    raise ValueError('molecule multiplicity must be 1 (use an explicit system)')
            except (IndexError, KeyError, ValueError) as exc:
                raise ValueError(f'{path}:{number}: {exc}') from exc
    if not atoms:
        raise ValueError(f'{path}: no explicit [ atoms ] section')
    for residue, attrs in graph.nodes(data=True):
        state = graph.degree(residue, weight='atomic_bonds')
        maximum = 2 if attrs['species'] == 'MPD' else 3
        if state > maximum:
            raise ValueError(f'{path}: residue {residue} exceeds functionality {maximum}')
        if attrs['declared_state'] is not None and state != attrs['declared_state']:
            raise ValueError(f'{path}: residue {residue}: label {attrs["residue_name"]} '
                             f'disagrees with {state} inter-residue atomic bonds')
        attrs['state'] = state
    return graph


def largest_molecule_dc3(molecules):
    """DC3 of the most-monomer molecule; size ties use the lowest molecule ID."""
    largest = min(molecules, key=lambda row: (-int(row['monomers']), int(row['molecule'])),
                  default=None)
    return float(largest['DC3']) if largest is not None else math.nan


def analyze_graph(graph, system):
    species = Counter(nx.get_node_attributes(graph, 'species').values())
    maximum = min(2 * species['MPD'], 3 * species['TMC'])
    bonds = graph.size(weight='atomic_bonds')
    status = ('no_capacity' if maximum == 0 else
              'below_50_percent' if 2 * bonds < maximum else
              'at_50_percent' if 2 * bonds == maximum else 'included')
    audit = dict(system=system, MPD=species['MPD'], TMC=species['TMC'],
                 bonds=bonds, residue_edges=graph.number_of_edges(), maximum_bonds=maximum,
                 bond_fraction=divide(bonds, maximum), status=status)
    if status != 'included':
        return audit, [], None
    molecules = []
    for component in nx.connected_components(graph):
        if len(component) == 1:  # MPD0 and TMC0 are ignored in molecule statistics.
            continue
        counts = dict.fromkeys(COUNTS, 0)
        for residue in component:
            attrs = graph.nodes[residue]
            key = ('M' if attrs['species'] == 'MPD' else 'T') + str(attrs['state'])
            counts[key] += 1
        molecules.append(dict(system=system, molecule=len(molecules) + 1,
                              residue_ids=' '.join(map(str, sorted(component))),
                              monomers=len(component), **counts, **measures(counts)))
    total = {key: sum(m[key] for m in molecules) for key in COUNTS}
    valid = [m for m in molecules if 0 <= m['DC3'] <= 1]
    summary = dict(**audit, molecules=len(molecules),
                   ignored_monomers=sum(a['state'] == 0 for _, a in graph.nodes(data=True)),
                   monomers=sum(m['monomers'] for m in molecules), **total, **measures(total),
                   largest_molecule_DC3=largest_molecule_dc3(molecules),
                   largest_molecule_monomers=max((m['monomers'] for m in molecules), default=0),
                   valid_DC3_molecules=len(valid),
                   mean_DC3=divide(sum(m['DC3'] for m in valid), len(valid)),
                   weighted_DC3=divide(sum(m['DC3'] * m['monomers'] for m in valid),
                                       sum(m['monomers'] for m in valid)))
    return audit, molecules, summary


def write_csv(path, rows, fields):
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def highest_bond_fraction_systems(systems):
    """Select every system tied for the maximum finite bond fraction per r folder."""
    maxima = {}
    for row in systems:
        folder = Path(row['system']).parts[0]
        fraction = float(row['bond_fraction'])
        if math.isfinite(fraction):
            maxima[folder] = max(maxima.get(folder, -math.inf), fraction)
    return [row for row in systems
            if float(row['bond_fraction']) == maxima.get(Path(row['system']).parts[0])]

def replot_tables(output):
    """Reuse analyzed counts, updating normalized columns without rereading topologies."""
    tables = []
    for filename in ('molecules.csv', 'systems.csv'):
        path = output / filename
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields = [key for key in reader.fieldnames
                      if key not in ('M1_T3', 'M2_T3', 'T1_T3', 'T2_T3', *RATIOS)]
            rows = []
            for raw in reader:
                row = {key: raw[key] for key in fields}
                for key in COUNTS:
                    row[key] = int(row[key])
                for key in ('bond_fraction', 'mean_DC3', 'weighted_DC3'):
                    if key in row:
                        row[key] = float(row[key])
                row.update(measures(row))
                rows.append(row)
        write_csv(path, rows, [*fields, *RATIOS])
        tables.append(rows)


def delete_candidates(output, candidates):
    """Delete verified inputs, preserving a durable planned/completed manifest."""
    # Check the whole deletion set before deleting any file.
    for path, size, mtime in candidates:
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != (size, mtime):
            raise RuntimeError(f'{path} changed during analysis; deletion aborted')
    # Append so re-running never destroys the historical deletion record.
    with (output / 'deleted_files.jsonl').open('a') as manifest:
        for path, size, mtime in candidates:
            manifest.write(json.dumps(dict(path=str(path.resolve()), bytes=size,
                                          mtime_ns=mtime, status='planned')) + '\n')
            manifest.flush()
            os.fsync(manifest.fileno())
            path.unlink()
            manifest.write(json.dumps(dict(path=str(path.resolve()), status='deleted')) + '\n')
            manifest.flush()
    print(f'Deleted {len(candidates)} topologies strictly below 50%.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('topology_results'))
    parser.add_argument('--delete-low-bond-files', action='store_true',
                        help='Delete topologies strictly below 50%% after successful output generation')
    args = parser.parse_args()
    paths = sorted(args.root.glob('*/*.top'))
    if not paths:
        parser.error('no */*.top files found')
    args.output.mkdir(parents=True, exist_ok=True)
    audit, molecules, systems, candidates, errors, unsupported = [], [], [], [], [], []
    for index, path in enumerate(paths, 1):
        system = path.relative_to(args.root).as_posix()
        try:
            before = path.stat()
            row, mols, summary = analyze_graph(read_graph(path), system)
            audit.append(row)
            molecules.extend(mols)
            if summary:
                systems.append(summary)
            if row['status'] == 'below_50_percent':
                candidates.append((path, before.st_size, before.st_mtime_ns))
        except UnsupportedTopology as exc:
            unsupported.append(dict(system=system, reason=str(exc)))
            audit.append(dict(system=system, MPD='', TMC='', bonds='', residue_edges='',
                              maximum_bonds='', bond_fraction='', status='unsupported'))
        except ValueError as exc:
            errors.append(dict(system=system, error=str(exc)))
        if index % 50 == 0 or index == len(paths):
            print(f'Analyzed {index}/{len(paths)}; included {len(systems)}; errors {len(errors)}', flush=True)
    audit_fields = ['system', 'MPD', 'TMC', 'bonds', 'residue_edges', 'maximum_bonds', 'bond_fraction', 'status']
    write_csv(args.output / 'audit.csv', audit, audit_fields)
    write_csv(args.output / 'errors.csv', errors, ['system', 'error'])
    write_csv(args.output / 'unsupported.csv', unsupported, ['system', 'reason'])
    write_csv(args.output / 'molecules.csv', molecules,
              ['system', 'molecule', 'residue_ids', 'monomers', *COUNTS, *METRICS, *RATIOS])
    write_csv(args.output / 'systems.csv', systems, [*audit_fields, 'molecules', 'ignored_monomers',
              'monomers', *COUNTS, *METRICS, *RATIOS, 'valid_DC3_molecules', 'mean_DC3', 'weighted_DC3',
              'largest_molecule_DC3', 'largest_molecule_monomers'])
    if errors:
        raise SystemExit(f'{len(errors)} invalid topologies; see errors.csv. No files deleted;')
    if args.delete_low_bond_files:
        delete_candidates(args.output, candidates)
    print(f'Retained {len(unsupported)} unsupported topologies; see unsupported.csv.', flush=True)
    print(f'Results: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    main()
