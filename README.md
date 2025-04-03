# Fine-tuning CLIP's Text Encoder for Linguistic Adaptability


## Overview

This repository contains the code for the blog post "Bridging the Gap: Enhancing CLIP's Linguistic Adaptability"

## Setup

The code can run with similar environment as [Open Clip](https://github.com/mlfoundations/open_clip), we also provided the  `environment.yml` file as reference

## Code Structure
- **config/**
  - `config.yaml` storing dataset paths
- **dataset_management/**
  - `dir_label_name_1k.json` a mapping for imagenet class names to their corresponding WordNet synset IDs
  - `open_image_noun_filtered.csv` a filtered version of the OpenImage dataset, containing the corresponding WordNet synset IDs
  - `train_image_ids.txt` the file for downloading OpenImage dataset 
- `environment.yml` reference file for setting up conda environment 
- **main/**
  - `classification_csv_maker.py` scripts to create a subset for Fer2013/ImageNet/OpenImage datasets to run experiment
  - `clip_classification.py` evaluation script for CLIP model that shows the impact of lexical adaptation
  - `training_utility.py` utility functions for training
  - `train_text_encoder.py` main script for training the text encoder
- `README.md`
- **zero_shot_analyze/**
  - **imagenet/**
    - `100_synonym_sample.csv` sample file for imagenet synonym task, for evaluation purpose
    - `100_synonym_sample.json` sample file for imagenet synonym task, for training purpose

## Dataset Preparation
### ImageNet
1. Download both the training set from [ImageNet official webiste](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php).
2. Set the "imagenet_root_path_train: " in `config/config.yaml` to the path of the downloaded training set.
3. Make sure that it is structured in following format
```├── n01440764
├── n01443537
├── n01484850
├── n01491361
......
```
### OpenImage
1. Follow the instruction on the [official website](https://storage.googleapis.com/openimages/web/download_v7.html) - "Download Manually" by using the file `train_image_ids.txt`, 
2. Insert the image path before the first column of `open_image_noun_filtered.csv`.
### FER2013
The dataset can be downloaded via [kaggle](https://www.kaggle.com/datasets/msambare/fer2013?resource=download) directly 
### More Dataset?
From our knowledge, there is no widely used Image-classification dataset that contains the synset IDs. 
If you have such dataset and want to test the method on it, please reference the format of the provided sample files and `classification_csv_maker.py` for dataset preparation.

## Usage
### Sample Run
You can do a trial run of the training script directly (without evaluation) by using the following command:
```bash
cd main
python train_text_encoder.py
```
- During the run, the script will print some information about the training process.  
- You should expect the Loss and Reg Loss be similar after the first few epochs.
- In the end, the script will save the trained model to the `tuned_models` directory (created automatically if it does not exist).

### Full-Pipeline
If the dataset is prepared, you can do a full pipeline as follows:
1. Select a function in `classification_csv_maker.py` to create a subset of the dataset:
   > **Note:** we usually suggest setting `level` to 1 since high level hypernyms are not very informative.  
   > `Number of classes` corresponds to the number of classes you want to use for the dataset. 
   >  If it is too large, the script will automatically select the largest possible non-overlapping subset for a given task
   - `imagenet_csv_synonym(number_of_classes=)` For ImageNet Synonym task
   - `imagenet_csv_hypernym(number_of_classes=, level=)` For ImageNet Hypernym task
   - `imagenet_mixed(number_of_classes=, level=)` For ImageNet Synonym + Hypernym task
   - `openimage_csv_synonym(number_of_classes=)` For OpenImage Synonym task
   - `openimage_csv_hypernym(number_of_classes=, level=)` For OpenImage Hypernym task
   - `fer_csv_synonym()` For FER2013 Synonym task

2. Change the variables `dataset_name` and `target_file_name` in `train_text_encoder.py` to the name of the dataset you want to use.

3. run the training script, the evaluation will be done automatically after the training
```bash
python train_text_encoder.py
```

