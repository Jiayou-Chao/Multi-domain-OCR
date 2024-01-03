# Multi-domain-OCR

The project is the codebase for the paper "Multi-domain OCR with Meta Self-Learning" (https://arxiv.org/abs/2401.00971). The code is based on [MMOCR](https://github.com/open-mmlab/mmocr).

## Installation

The environment setup is the same as [MMOCR](https://mmocr.readthedocs.io/en/dev-1.x/get_started/install.html). Alternatively you can use the `setup.sh` script to install the environment:

```bash
# (Optional) Create a conda environment
conda create -n multi-domain-ocr python=3.10 -y 
conda activate multi-domain-ocr

# Set up the environment
bash setup.sh
```

## Prepare Dataset

The dataset used for evaluation is the open-sourced dataset [MSDA (Multi-source domain adaptation dataset for text recognition)](https://bupt-ai-cz.github.io/Meta-SelfLearning/). Please refer to the homepage for the download link. The dataset is in the format of `tar` file. Please extract the file and then use `tools/dataset_converters/textrecog/lmdb_converter.py` to convert the dataset to `lmdb` format. Assume the dataset is stored in `data/Meta-SelfLearning` and to be extracted to `data/cache/`, the following is an example of converting the `syn` dataset to `lmdb` format:

```bash
# Extract the dataset
mkdir -p data/cache/
tar -xvf data/Meta-SelfLearning/syn/test_imgs.tar -C data/cache/

# Convert the dataset to lmdb format
python tools/dataset_converters/textrecog/lmdb_converter.py data/Meta-SelfLearning/syn/test_label.txt data/Meta-SelfLearning/LMDB/syn/test_imgs.lmdb -i data/cache/Meta-SelfLearning/root/data/TextRecognitionDatasets/IMG/syn/test_imgs/ --label-format txt
```

## Training

The training script is in `tools/train.py`. The following is an example of training the model on the `syn` dataset:

```bash
python tools/train.py configs/path/to/config.py
```

If you want to use multiple GPUs for training, use `tools/dist_train.sh`:
    
```bash
tools/dist_train.sh configs/configs/path/to/config.py 8 --auto-scale-lr --amp
```

The config files to reproduce the results in the paper are in `configs/`. The following is an example of training the backbone on the `docu` dataset:

```bash
tools/dist_train.sh configs/textrecog/adapter/backbone_docu.py 8 --auto-scale-lr --amp
```

The following is an example of training the adapter on the `syn` dataset:

```bash
tools/dist_train.sh configs/textrecog/adapter/adapter_docu_adapter_syn.py 8 --auto-scale-lr --amp
```

