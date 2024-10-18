<p align="center">
  <h1 align="center"> <ins>Descriptor Reasoning</ins><br> Leveraging Semantic Cues from Foundation Vision Models <br> for Enhanced Local Feature Correspondence <br>⭐ACCV 2024⭐</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?">Felipe Cadar</a>
    ·
    <a href="https://scholar.google.com/citations?">Guilherme Potje</a>
    ·
    <a href="https://scholar.google.com/citations?">Renato Mastins</a>
    ·
    <a href="https://scholar.google.com/citations?">Cédric Demonceaux</a>
    ·
    <a href="https://scholar.google.com/citations?">Erickson R Nascimento</a>
  </p>
  <h2 align="center"><p>
    <!-- <a href="https://arxiv.org/abs/2305.15404" align="center">Paper</a> -->
    <!-- <a href="https://parskatt.github.io/RoMa" align="center">Project Page</a> -->
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<!-- <p align="center">
    <img src="https://github.com/Parskatt/RoMa/assets/22053118/15d8fea7-aa6d-479f-8a93-350d950d006b" alt="example" width=80%>
    <br>
    <em>RoMa is the robust dense feature matcher capable of estimating pixel-dense warps and reliable certainties for almost any image pair.</em>
</p> -->

## Installation

To set up the environment for training, run the following command to create a new conda environment with Python 3.9:
```bash
conda create -n reason  python=3.9
```
Activate the environment before proceeding:
```bash
conda activate reason
```

Install the package:
```bash
pip install -e .
```

# Inference

# Training 

## Data Preparation for Training

### Scannet Data Preparation

To prepare the Scannet dataset for training, follow these steps:

1. **Download Scannet**: First, download the Scannet dataset. Make sure to read and accept the terms of use.
```bash
python reasoning/scripts/scannet/01_download_scannet.py --out_dir datasets/scannet
```
2. **Extract Frames**: Extract frames from the downloaded dataset, skipping every 15 frames.
```bash
python reasoning/scripts/scannet/02_extract_scannet.py --data_path datasets/scannet
```
3. **Calculate Covisibility**: Calculate the covisibility between frames to identify good pairs for training.
```bash
python reasoning/scripts/scannet/03_calculate_scannet_covisibility.py --data_path datasets/scannet
```
4. **Convert to H5 Files**: Convert the prepared data into H5 files for easier handling during training. It also helps to keep the number of files small in cluster enviroments. 

```bash
python reasoning/scripts/scannet/04_build_h5.py --data_path datasets/scannet --output datasets/h5_scannet/
```

### Feature Extraction

To speed up the training process, pre-extract some features from the dataset. Ours scripts read the h5 dataset and save the features to the save directory

#### DINOv2-S Features Extraction

Extract DINOv2-S features from the H5 dataset. You can adjust the batch size according to your system's capabilities.
```bash
python reasoning/scripts/export_dino.py --data ./datasets/h5_scannet --batch_size 4 --dino_model dinov2_vits14
```
For larger models, simply change the `--dino_model` argument to one of the following: `dinov2_vitb14`, `dinov2_vitl14`, or `dinov2_vitg14`.

#### XFeat Features Extraction

Extract XFeat features from the dataset. Adjust the batch size as needed.
```bash
python reasoning/scripts/export_xfeat.py --data ./datasets/h5_dataset --batch_size 4 --num_keypoints 2048 h5_scannet
```
Your dataset folder should look like this:
```
datasets/
├── h5_scannet/
│   ├── train/
│   ├── features/
│   │   ├── dino-scannet-dinov2_vits14/
│   │   └── xfeat-scannet-n2048/
└── scannet/
    └── scans/
```

For other descriptors, please check the `reasoning/scripts/export_*.py` scripts.


## Training the Model

All training and experiments were conducted on a SLURM cluster with 4xV100 32GB GPUs. Adjust the batch size to match your system's capabilities.

To start training, run the following command:
```bash
python reasoning/train_multigpu_reasoning.py \
    --batch_size 2 \ 
    --data ./datasets/h5_scannet \ # dataset folder with images and features
    --plot_every 200 \ # tensorboard matching plots
    --extractor_cache 'xfeat-scannet-n2048' \ # local features
    --dino_cache 'dino-scannet-dinov2_vits14' \ # semantic features
    -C xfeat-dinov2 # comment for tracking your exps
```

## Acknowledgements
This work was partially supported by grants from CAPES, CNPq, FAPEMIG, Google, ANER MOVIS from Conseil Régional BFC and ANR (ANR-23-CE23-0003-01), to whom we are grateful. This project was also provided with AI computing and storage resources by GENCI at IDRIS thanks to the grant 2024-AD011015289 on the supercomputer Jean Zay’s V100 partitions.

We thank the authors of [DeDoDe](https://github.com/Parskatt/DeDoDe) for providing a template for this readme file.