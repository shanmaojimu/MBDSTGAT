# MBDSTGAT
This repository includes the code for MBDSTGAT, baseline models, comparison experiments, ablation experiments, online experiments, and some plotting code. Due to the large size of the log files during the experimental process, we have only retained the logs for MBDSTGAT. If researchers need to refer to the baseline model logs, they can contact the author via email.
## Config
python version: 3.10.2 <br>
pyTorch version: 2.5.1+cpu <br>
braindecode version: 0.8 <br>
mne version: 1.9.0 <br>
pandas version: 2.2.2 <br>
numPy version: 1.24.3 <br>
matplotlib version: 3.9.2 <br>
scikit-learn version: 1.7.2 <br>
seaborn version: 0.13.2 <br>
thop version: 0.1.1.post2209072238<br>
scipy version: 1.15.3 <br>
Computations performed — NVIDIA GeForce RTX 3090 GPU <br>

## Initial Parameter Settings

python train_MBDSTGAT.py     -device 0     -data_type HGD     -data_path /yours     -id 1     -father_path saved_results     -spatial_GAT True       -time_GAT True     -use_kgat True  -window_length 125  -window_padding 100  -epochs 2000 -batch_size 32 -lr 2 ** -12 <br>
<br>
python train_MBDSTGAT.py     -device 1     -data_type bci2a     -data_path /yours     -id 1     -father_path saved_results     -spatial_GAT True     -time_GAT True     -use_kgat True  -window_length 125  -window_padding 100  -epochs 2000 -batch_size 32 -lr 2 ** -12 <br>
<br>
python train_MBDSTGAT.py     -device 3     -data_type VR-MI     -data_path /yours     -id 1     -father_path saved_results     -spatial_GAT True     -time_GAT True     -use_kgat True  -window_length 125  -window_padding 100  -epochs 2000 -batch_size 32 -lr 2 ** -12 <br>
<br>
1、MTP 
-use_adj_perturbation True
| Parameter Name | Type | Default Value | Description |
|--------|------|--------|------|
| `-adj_edge_change_ratio` | float | 0.1 | Edge Change Ratio (0.0-1.0) |
| `-adj_weight_noise_std` | float | 0.1 | Weight Noise Standard Deviation |
| `-adj_min_edge_weight` | float | 0.01 | Minimum Edge Weight |
| `-adj_max_edge_weight` | float | 1.0 | Maximum Edge Weight |
<br>
2、kgat
-use_kgat Ture <br>

## VR-MI Dataset
Due to the large size of the self-collected VR-MI dataset, it is not publicly available in this repository. However, we have made the dataset publicly available on IEEE DataPort with [doi: 10.21227/89ez-a093](https://dx.doi.org/10.21227/89ez-a093).
