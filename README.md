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
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
python -m pip install -r requirements.txt
```

# Usage

## Step 1: prepare single-cell read counts data

We use same pipeline as used in [rcCAE](https://github.com/zhyu-lab/rccae) to prepare the single-cell read counts data, please refer to [rcCAE](https://github.com/zhyu-lab/rccae) for detailed instructions. 

## Step 2: Imputation and denoising of read counts

The “train.py” Python script is used to train the model and get reconstructed read counts data.

The arguments to run “train.py” are as follows:

| Parameter     | Description                                                   | Possible values                   |
| ------------- | ------------------------------------------------------------- | --------------------------------- |
| --input       | input file containing single-cell read counts                 | Ex: /path/to/example.txt          |
| --output      | a directory to save results                                   | Ex: /path/to/results              |
| --epochs      | number of epoches to train the scTCA                          | Ex: epochs=300  default:200       |
| --batch_size  | batch size                                                    | Ex: batch_size=64  default:32     |
| --lr          | learning rate                                                 | Ex: lr=0.0005  default:0.0001     |
| --latent_dim  | the latent dimension                                          | Ex: latent_dim=10  default:5      |
| --max_seg_len | the maximum length of subsequence for stepwise self attention | Ex: max_seg_len=1000  default:500 |
| --seed        | random seed (for reproduction of the results)                 | Ex: seed=1  default:0             |

Example:

```bash
tar -zxvf data/A_50k.tar.gz
python train.py --input ./data/A_50k.txt --epochs 100 --batch_size 32 --lr 0.0001 --latent_dim 5 --seed 0 --output data
```

# Reproduce the results

The instructions to reproduce results of scTCA on real datasets (take dataset A as an example) are provided as follows.

```bash
#BAM file (breast_tissue_A_2k_possorted_bam.bam) of dataset A can be downloaded from https://www.10xgenomics.com/datasets/breast-tissue-nuclei-section-a-2000-cells-1-standard-1-1-0
#hg19 reference file (hg19.fa.gz) can be downloaded from https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips
#mappability file (wgEncodeCrgMapabilityAlign36mer.bigWig) can be downloaded from https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeMapability/
#step 1: get read counts from the BAM file
./rccae/prep/bin/prepInput -b breast_tissue_A_2k_possorted_bam.bam -r hg19.fa -m wgEncodeCrgMapabilityAlign36mer.bigWig -B ./data/barcode_A.filtered.txt -c 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 -s 20000 -o ./data/A_20k.txt
#step 2: perform imputation and data smoothing
mkdir ./results
python train.py --input ./data/A_20k.txt --epochs 100 --batch_size 32 --lr 0.0001 --latent_dim 5 --seed 0 --output ./results
```

# Contact

If you have any questions, please contact lfr_nxu@163.com.