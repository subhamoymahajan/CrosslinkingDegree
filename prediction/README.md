# DC3 prediction

This folder is portable: keep `predict_dc3.py` and the two `.keras` models together.
No training data or files outside this folder are needed.

Install dependencies with `python -m pip install -r requirements.txt`.
The existing `tfml` conda environment can also run the script.

```sh
python predict_dc3.py example_1D.csv --model 1D -o results_1D.csv
python predict_dc3.py example_2D.csv --model 2D -o results_2D.csv
```

CSV headers must be `DP,O/N` for 1D or `DP,O/N,O2/O` for 2D.
DP means the model's bond fraction, from 0 to 1 (not a percentage).
O/N must be positive; O2/O must be between 0 and 1. All inputs must
be finite numbers. Invalid rows cause an error before output is written.
Extra columns and input row order are preserved. The output adds
`predicted_DC3` (replacing that column if already present). Without `-o`,
output is saved beside the input as `<input_stem>_DC3_<model>.csv`.
Models include normalization; do not standardize the inputs yourself.

The models are the folds with lowest held-out RMSE for each input family,
matching the existing NC prediction scripts: fold 8 for both 1D and 2D.
The 1D network has hidden layers 16/16 (RMSE 0.0222314); the 2D network
has hidden layers 64/16 (RMSE 0.0197482). These are selection scores,
not independent performance estimates. `model_metadata.json` records
source paths, feature order, scores, and SHA-256 checksums.

## Convert measurement CSVs

`conversions.py` supports the three existing forward conversions to 2D XPS
ratios. It appends (or replaces) `O/N` and `O2/O`, preserving other columns,
including `DP`, so the result can be passed to `predict_dc3.py`.

```sh
python conversions.py ftir-nmr input.csv -o converted.csv
python conversions.py ftir-nmr input.csv --dp 0.9 -o converted.csv
python conversions.py monomer input.csv -o converted.csv
python conversions.py song input.csv --columns 'C (%)' 'O (%)' 'N (%)' -o converted.csv
python predict_dc3.py converted.csv --model 2D -o predictions.csv
```

Default input headers are `ftir,nmr`, `m1,m2,t1,t2,t3`, and `C,O,N`,
respectively. Use `--columns` to specify different headers in that order.
FTIR/NMR inputs are dimensionless ratios; monomer populations must share a
common scale; Song inputs are atomic amounts/fractions/percentages, not mass
percentages. Without `-o`, output is `<input_stem>_2D_xps.csv` beside the input.
Missing/non-numeric/nonfinite inputs and undefined conversions fail before
writing output. Finite ratios outside predictor limits are reported and
preserved without clipping. If the input has no `DP` column, conversions add
`DP=1`. An existing `DP` column is preserved unless `--dp VALUE` is supplied,
which overrides DP for every row. The constant must be finite and between
0 and 1. No inverse conversions are implemented.

## Comparison with literature

The experimental data provided in Section S8 of the supporting information: `exp_karan.csv` and `exp_monomers.csv`. 


