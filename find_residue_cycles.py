#!/usr/bin/env python3
"""Find a NetworkX cycle basis in an explicit MPD/TMC residue topology."""
import argparse
import csv
import json
from pathlib import Path

import networkx as nx

from analyze_topology import read_graph


def atom_records(path):
    """Read topology atom order, which must match the VMD coordinate atom order."""
    records = []
    section = ''
    for line in path.read_text().splitlines():
        line = line.split(';', 1)[0].strip()
        if line.startswith('['):
            section = line.split(']', 1)[0][1:].strip()
        elif line and section == 'atoms':
            fields = line.split()
            records.append(dict(atom_id=int(fields[0]), residue_id=int(fields[2]),
                                residue_name=fields[3], atom_name=fields[4],
                                vmd_index=len(records)))
    return records


def atomic_connectivity(path, atoms):
    """Read actual bonds, keeping sorted adjacency for deterministic shortest paths."""
    by_id = {a['atom_id']: a for a in atoms}
    edges = set()
    section = ''
    for line in path.read_text().splitlines():
        line = line.split(';', 1)[0].strip()
        if line.startswith('['):
            section = line.split(']', 1)[0][1:].strip()
        elif line and section == 'bonds':
            a, b = map(int, line.split()[:2])
            if a not in by_id or b not in by_id:
                raise ValueError(f'Bond references an unknown atom: {a}, {b}')
            edges.add(tuple(sorted((a, b))))
    atomic = nx.Graph()
    atomic.add_nodes_from(sorted(by_id))
    atomic.add_edges_from(sorted(edges))
    interiors, links = {}, {}
    for atom in atoms:
        interiors.setdefault(atom['residue_id'], nx.Graph()).add_node(atom['atom_id'])
    for a, b in sorted(edges):
        ra, rb = by_id[a]['residue_id'], by_id[b]['residue_id']
        if ra == rb:
            interiors[ra].add_edge(a, b)
        else:
            links.setdefault((ra, rb), []).append((a, b))
            links.setdefault((rb, ra), []).append((b, a))
    return atomic, interiors, links, by_id


def atomic_ring_path(residues, interiors, links):
    # Pick the lowest atom-ID pair when multiple links connect a residue pair.
    crosslinks = [min(links[(a, b)]) for a, b in
                  zip(residues, residues[1:]+residues[:1])]
    path = []
    for i, residue in enumerate(residues):
        entry, departure = crosslinks[i-1][1], crosslinks[i][0]
        try:
            path.extend(nx.shortest_path(interiors[residue], entry, departure))
        except nx.NetworkXNoPath as exc:
            raise ValueError(f'No internal bonded path in residue {residue}') from exc
    return path


def canonical_cycle(cycle):
    start = cycle.index(min(cycle))
    forward = cycle[start:]+cycle[:start]
    reverse = [forward[0]]+forward[:0:-1]
    return min(forward, reverse)


def basis_cycles(graph):
    """Use sorted insertion and an explicit root per component for reproducibility."""
    ordered = nx.Graph()
    ordered.add_nodes_from(sorted(graph.nodes))
    ordered.add_edges_from(sorted(tuple(sorted(edge)) for edge in graph.edges))
    cycles = []
    components = sorted(nx.connected_components(ordered), key=lambda c: min(c))
    for component_id, nodes in enumerate(components, 1):
        subgraph = ordered.subgraph(nodes)
        basis = nx.cycle_basis(subgraph, root=min(nodes))
        for cycle in sorted((canonical_cycle(c) for c in basis), key=lambda c: (len(c), c)):
            cycles.append(dict(cycle_id=len(cycles)+1, component_id=component_id,
                               size=len(cycle), residue_ids=cycle))
    expected = graph.number_of_edges()-graph.number_of_nodes()+len(components)
    if len(cycles) != expected:
        raise RuntimeError('Cycle basis dimension disagrees with E - V + C.')
    return cycles


def tcl_word(value):
    # Topology labels are data, never executable Tcl source.
    return '"'+str(value).replace('\\', '\\\\').replace('"', '\\"').replace('$', '\\$').replace('[', '\\[').replace(']', '\\]').replace('\n', '\\n')+'"'


def analyze(topology, output):
    graph = read_graph(topology)
    atoms = atom_records(topology)
    cycles = basis_cycles(graph)
    atomic, interiors, links, by_id = atomic_connectivity(topology, atoms)
    residue_atoms = {}
    for atom in atoms:
        if atom['residue_id'] in graph:
            residue_atoms.setdefault(atom['residue_id'], []).append(atom['vmd_index'])
    memberships = {residue: [] for residue in graph}
    for cycle in cycles:
        cycle['residue_names'] = [graph.nodes[r]['residue_name'] for r in cycle['residue_ids']]
        cycle['residue_vmd_indices'] = sorted(i for r in cycle['residue_ids'] for i in residue_atoms[r])
        path = atomic_ring_path(cycle['residue_ids'], interiors, links)
        if len(set(path)) != len(path) or not all(atomic.has_edge(a, b) for a, b in
                                                 zip(path, path[1:]+path[:1])):
            raise ValueError('Invalid atom-level closed ring path')
        cycle['ordered_atom_ids'] = path
        cycle['ordered_vmd_indices'] = [by_id[a]['vmd_index'] for a in path]
        cycle['vmd_indices'] = sorted(cycle['ordered_vmd_indices'])
        for r in cycle['residue_ids']:
            memberships[r].append(cycle['cycle_id'])
    report = dict(topology=str(topology.resolve()), networkx_version=nx.__version__,
                  method='networkx.cycle_basis, sorted graph, minimum residue root per component',
                  scope='Simple MPD/TMC residue graph; no intramonomer rings or parallel-edge cycles',
                  note='A basis is not all simple cycles or a minimum cycle basis.',
                  atom_path='Shortest bonded path inside each residue; sorted adjacency breaks ties; lowest atom-ID crosslink pair if multiple',
                  nodes=len(graph), edges=graph.number_of_edges(),
                  components=nx.number_connected_components(graph), cycle_rank=len(cycles),
                  atoms=len(atoms), cycles=cycles)
    output.mkdir(parents=True, exist_ok=True)
    (output/'cycles.json').write_text(json.dumps(report, indent=2)+'\n')
    with (output/'cycles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['cycle_id', 'component_id', 'size', 'ordered_residue_ids', 'residue_names'])
        for c in cycles:
            writer.writerow([c['cycle_id'], c['component_id'], c['size'],
                             ' '.join(map(str, c['residue_ids'])), ' '.join(c['residue_names'])])
    with (output/'ring_residues.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['residue_id', 'residue_name', 'basis_cycle_count', 'cycle_ids', 'vmd_indices'])
        for r, ids in sorted(memberships.items()):
            if ids:
                writer.writerow([r, graph.nodes[r]['residue_name'], len(ids),
                                 ' '.join(map(str, ids)), ' '.join(map(str, residue_atoms[r]))])
    # Bundle the viewer and numeric selection data into one directly sourceable file.
    viewer = Path(__file__).with_name('vmd_ring_viewer.tcl').read_text()
    data = ['namespace eval ::rings {', f'    variable expected_atoms {len(atoms)}',
            '    variable expected_names [list '+ ' '.join(tcl_word(a['atom_name']) for a in atoms)+']',
            '    variable cycles [dict create]', '    variable atom_paths [dict create]']
    for c in cycles:
        data.append(f'    dict set cycles {c["cycle_id"]} {{{" ".join(map(str, c["residue_ids"]))}}}')
        data.append(f'    dict set atom_paths {c["cycle_id"]} {{{" ".join(map(str, c["ordered_vmd_indices"]))}}}')
    data += ['}', viewer]
    (output/'view_rings.tcl').write_text('\n'.join(data)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topology', type=Path, help='Explicit single-moleculetype .top file')
    parser.add_argument('--output', type=Path, help='Default: topology_results/cycles/<folder>/<stem>')
    args = parser.parse_args()
    output = args.output or Path('topology_results/cycles')/args.topology.parent.name/args.topology.stem
    try:
        report = analyze(args.topology, output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'{report["nodes"]} residues, {report["edges"]} edges, '
          f'{report["components"]} components, {report["cycle_rank"]} basis cycles')
    print(f'Saved {output}/cycles.json, cycles.csv, ring_residues.csv, view_rings.tcl')
    print('Load matching coordinates in VMD, source view_rings.tcl, then: rings::show 1')


if __name__ == '__main__':
    main()
