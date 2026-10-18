"""Standalone DC3 inference: python predict_dc3.py input.csv --model 1D.

Input columns: DP, O/N, and O2/O for 2D. DP is bond fraction (0 to 1).
Saved Keras models include their own input normalization.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
import pandas as pd
import tensorflow as tf


def predict(input_path, output_path, model_name):
    input_path, output_path = Path(input_path), Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError('Output must differ from the input CSV.')
    data = pd.read_csv(input_path)
    columns = ['DP', 'O/N'] + (['O2/O'] if model_name == '2D' else [])
    missing = [column for column in columns if column not in data]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if data.empty:
        raise ValueError('Input CSV has no data rows.')
    values = data[columns].apply(pd.to_numeric, errors='coerce')
    valid = (np.isfinite(values).all(axis=1) & values.DP.between(0, 1)
             & (values['O/N'] > 0))
    if model_name == '2D':
        valid &= values['O2/O'].between(0, 1)
    if not valid.all():
        rows = (data.index[~valid] + 2).tolist()
        raise ValueError(f'Invalid inputs at CSV rows {rows}: require finite numbers, '
                         '0 <= DP <= 1, O/N > 0, and 0 <= O2/O <= 1 for 2D.')
    model_path = Path(__file__).resolve().parent / f'FFNN_{model_name}.keras'
    model = tf.keras.models.load_model(model_path, compile=False)
    if model.input_shape[-1] != len(columns):
        raise ValueError(f'Model input shape does not match {columns}')
    predictions = model(values.to_numpy(dtype=np.float32), training=False).numpy().reshape(-1)
    if len(predictions) != len(data) or not np.isfinite(predictions).all():
        raise ValueError('Model returned invalid predictions.')
    data['predicted_DC3'] = predictions
    data.to_csv(output_path, index=False)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_csv', type=Path)
    parser.add_argument('--model', type=str.upper, choices=['1D', '2D'], required=True)
    parser.add_argument('-o', '--output', type=Path,
                        help='Default: <input_stem>_DC3_<model>.csv beside input')
    args = parser.parse_args()
    output = args.output or args.input_csv.with_name(f'{args.input_csv.stem}_DC3_{args.model}.csv')
    try:
        count = predict(args.input_csv, output, args.model)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Wrote {count} {args.model} predictions to {output}')


if __name__ == '__main__':
    main()
