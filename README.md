# scTCA
Imputation and denoising of scDNA-seq data

## Requirements

* Python 3.9+.

# Installation

## Clone repository

First, download scTCA from github and change to the directory:

```bash
git clone https://github.com/zhyu-lab/sctca
cd sctca
```

## Create conda environment (optional)

Create a new environment named "sctca":

```bash
conda create --name sctca python=3.9
```

Then activate it:

```bash
conda activate sctca
```

## Install requirements

```bash
python -m pip install -r requirements.txt
```

# Usage

## Step 1: prepare single-cell read counts data

We use same pipeline as used in [rcCAE](https://github.com/zhyu-lab/rccae) to prepare the single-cell read counts data, please refer to [rcCAE](https://github.com/zhyu-lab/rccae) for detailed instructions. 

## Step 2: Imputation and denoising of read counts

The “train.py” Python script is used to train the model and get reconstructed read counts data.

The arguments to run “train.py” are as follows:

| Parameter      | Description                                                   | Possible values                    |
| -------------- | ------------------------------------------------------------- | ---------------------------------- |
| --input        | input file containing single-cell read counts                 | Ex: /path/to/example.txt           |
| --output       | a directory to save results                                   | Ex: /path/to/results               |
| --epochs       | number of epoches to train the scTCA                          | Ex: epochs=300  default:200        |
| --batch_size   | batch size                                                    | Ex: batch_size=64  default:32      |
| --lr           | learning rate                                                 | Ex: lr=0.0005  default:0.0001      |
| --latent_dim   | the latent dimension                                          | Ex: latent_dim=10  default:5       |
| --max_seg_len  | the maximum length of subsequence for stepwise self attention | Ex: latent_dim=4  default:10       |
| --seed         | random seed (for reproduction of the results)                 | Ex: seed=1  default:0              |

Example:

```
tar -zxvf data/example.tar.gz
python train.py --input A_50k.txt --epochs 100 --batch_size 32 --lr 0.0001 --latent_dim 5 --seed 0 --output data
```

# Contact

If you have any questions, please contact lfr_nxu@163.com.