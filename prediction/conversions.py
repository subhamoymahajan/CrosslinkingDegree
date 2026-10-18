"""Convert CSV measurements to the O/N and O2/O inputs of the 2D model.

Examples:
    python conversions.py ftir-nmr measurements.csv -o converted.csv
    python conversions.py monomer populations.csv
    python conversions.py song elemental.csv --columns 'C (%)' 'O (%)' 'N (%)'

These are the three forward conversions defined in this module; inverse
conversions are not inferred. Input columns and row order are preserved.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert_ftir_nmr_to_2D_xps(ftir, nmr):
    """Return (O/N, O2/O) from dimensionless FTIR and NMR ratios.

    Supply the ratios used by the original formula, not percentage values.
    The expressions are singular at nmr=0.5 or ftir=-1.5/-1.
    Scalars and NumPy/Pandas arrays are supported.
    """
    # Infer elemental O/N and the fraction of oxygen assigned to O2.
    rON = 3*nmr*(1+2*ftir)/((1-2*nmr)*(1+ftir)) 
    rO2 = 1/(2*(1+ftir))
    return rON, rO2 

def convert_monomer_to_2D_xps(m1, m2, t1, t2, t3):
    """Return (O/N, O2/O) from five monomer populations on a common scale.

    m1/m2 and t1/t2/t3 retain the species definitions of the original formula.
    Counts or fractions may be used because a common scale cancels out.
    """
    # Weighted oxygen population divided by the nitrogen population.
    rON = (5*t1+4*t2+3*t3)/(2*(m1+m2))
    # O2 population divided by the total oxygen population.
    rO2 =  (t1+2*t2+3*t3)/(5*t1+4*t2+3*t3)
    return rON, rO2


def convert_song_to_2D_xps(C, O, N):
    """Return (O/N, O2/O) using the existing Song composition relation.

    C, O, N must be atomic amounts, fractions, or percentages on the same
    scale, not mass percentages. Their sum, O, and N must be nonzero.
    """
    # Normalize to atomic fractions before applying the Song relation.
    rO = O/(C+O+N)
    rN = N/(C+O+N)
    rO2 = 2/(3*rO) - 8*rN/(3*rO ) -5/3
    return O/N, rO2


# Default source headers also define the positional order for --columns.
CONVERSIONS = {
    'ftir-nmr': (convert_ftir_nmr_to_2D_xps, ['FTIR', 'NMR']),
    'monomer': (convert_monomer_to_2D_xps, ['M1', 'M2', 'T1', 'T2', 'T3']),
    'song': (convert_song_to_2D_xps, ['C', 'O', 'N']),
}


def convert_csv(input_path, output_path, method, columns=None, dp=None):
    """Convert every row, rejecting missing/nonfinite data and singularities.

    Computed ratios are not clipped to physical/model limits. This preserves
    the formulas and allows inconsistent measurements to remain visible.
    Existing O/N and O2/O columns are replaced by the calculated values.
    A supplied dp overrides every row; otherwise an existing DP column is
    preserved, or DP=1 is added when the column is absent.
    """
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Output must differ from the input CSV.')
    if dp is not None and (not np.isfinite(dp) or not 0 <= dp <= 1):
        raise ValueError('Constant DP must be finite and between 0 and 1.')
    function, defaults = CONVERSIONS[method]
    columns = defaults if columns is None else columns
    if len(columns) != len(defaults):
        raise ValueError(f'{method} requires {len(defaults)} columns in order: {defaults}')
    data = pd.read_csv(input_path)
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f'Missing source columns: {missing}; use --columns to map headers.')
    if data.empty:
        raise ValueError('Input CSV has no data rows.')
    values = data[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    valid = np.isfinite(values).all(axis=1)
    if not valid.all():
        raise ValueError(f'Non-numeric, missing, or nonfinite input at CSV rows '
                         f'{(np.flatnonzero(~valid)+2).tolist()}')
    # Vectorized evaluation keeps each output aligned with its original row.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ron, ro2 = function(*values.T)
    finite = np.isfinite(ron) & np.isfinite(ro2)
    if not finite.all():
        raise ValueError(f'Undefined conversion (zero denominator or overflow) at CSV rows '
                         f'{(np.flatnonzero(~finite)+2).tolist()}')
    data['O/N'], data['O2/O'] = ron, ro2
    # Explicit constant takes precedence over per-row DP and the default.
    if dp is not None:
        data['DP'] = dp
    elif 'DP' not in data:
        data['DP'] = 1.0
    data.to_csv(output_path, index=False)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('method', choices=CONVERSIONS)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('-o', '--output', type=Path,
                        help='Default: <input_stem>_2D_xps.csv beside the input')
    parser.add_argument('--columns', nargs='+', metavar='HEADER',
                        help='Source headers in formula order: ftir nmr; '
                             'M1 M2 T1 T2 T3; or C O N')
    parser.add_argument('--dp', type=float, metavar='DP',
                        help='Constant DP (0 to 1), overriding all rows. Otherwise '
                             'preserve the DP column, or use 1 if absent.')
    args = parser.parse_args()
    output = args.output or args.input_csv.with_name(f'{args.input_csv.stem}_2D_xps.csv')
    try:
        data = convert_csv(args.input_csv, output, args.method, args.columns, dp=args.dp)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Wrote {len(data)} converted rows to {output}')
    outside = (data['O/N'] <= 0) | ~data['O2/O'].between(0, 1)
    if outside.any():
        print(f'Note: {int(outside.sum())} rows fall outside predictor input limits '
              '(O/N > 0 and 0 <= O2/O <= 1); values were retained without clipping.')


if __name__ == '__main__':
    main()

