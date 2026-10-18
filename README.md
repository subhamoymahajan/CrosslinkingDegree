# Requirements of Use

The data provided here are based on the following Manuscript: . Please cite the work, if you find the data provided here useful.

# Predicting DC3

Follow README in predictions/ folder, which has trained Feed-Forward Neural Networks needed for predictions. Data can be converted between different representations before predicting.

# Reproducing the Results in the Manuscript

## 1. Extract Data

`tar -xzvf Data.tar.gz` (the TAR file is located in Zenodo: )

`NC*.csv` data files are based on supporting information of https://doi.org/10.1038/s41467-024-45918-4 please cite them.

## 2. Analyze existing topology

Topology is provided only when DP > 0.5, 

`python3 analyze_topology.py`

Generate results and analysis in `topology_results/`.

The most important output is `topology_results/systems.csv`

## 3. Train Feed-Forward Neural Network 

Train network for predicing DC3 from O/N ratio and DP. Outputs are generated in `neural_network_results/*`. Parity plots are generated `dc3_parity.png` and performance metrics `metrics.json`. Model is saved in `dc3_network.keras`

`python train_neural_network.py`

Train network for predicting DC3 from O/N ratio, O2/O ratio, and DP. Outputs are generated in `neural_network_results_o2_o/*`.  Parity plots are generated `dc3_parity.png` and performance metrics `metrics.json`. Model is saved in `dc3_network.keras`.

`python train_neural_network.py --`

Hyper-parameters tuning for predicting DC3 from O/N ratio and DP. Outputs are generated in `neural_network_tining/*`

`python tune_neural_network.py --two-inputs`

Hyper-parameter tuning for predicting DC3 from O/N ratio, O2/O ratio, and DP. Outputs are generated in `neural_network_tuning_o2_o/*`

`python tune_neural_network.py`

Run k-fold cross-validation 
`python cross_validate_networks.py`

## 4. Plots 

**Plot k-fold cross validation errors and parity plots**
`python plot_cross_validation.py`
`python plot_dc3_parity.py`

**Plot DC3 Fig. 2a.**
`python plot_DC3_ON.py --no-show --no-colorbar`

**Plot DC3 Fig. 2b.**
`python plot_dc3_3d_surface.py`

**Plot Fig. 2c.**
Run prediction: `python predict_nc_1d_dc3.py ; python predict_nc_2d_dc3.py `

**Plot Fig. 3**
`python plot_combined_correlations.py`

**Plot Fig. 4b**
`python plot_ring_size_distribution.py r1.5/clink_26_0.top`

OR

`python plot_ring_size_distribution.py Equilibrated_Rings/r1.5/clink_26_0.top`

## 5. Equations 

The supporting information equations can be verified using jupyter notebooks provided `Equations.ipynb`, which uses `sympy`. 
