# STAS_Final
## Description
This is a competition from [AI CUP 2022](https://tbrain.trendmicro.com.tw/Competitions/Details/22). The goal of this competition is to segment spread through air spaces (STAS) from a lung adenocarcinoma H&E image.

## Requirements
This repository is implemented by `MATLAB2021a` and `python3.8.0` with third party package that stored in `requirements.txt`.

## Dataset and Folder structure
The dataset can be downloaded from [here](https://tbrain.trendmicro.com.tw/Competitions/Details/22). </br>
* For **reproduce our submission result only**, you may just download `Private_Image.zip` and `Public_Image.zip`.
* For **training scheme**, you may need to download `SEG_Train_Dataset.zip` also.

For convinence, you can follow the folder structure as follows
```
.
├─Dataset
|   ├─Private_Image      # From Private_Image.zip
|   ├─Public_Image       # From Private_Image.zip
|   ├─Train_Annotations  # From SEG_Train_Dataset.zip
|   └─Train_Images       # From SEG_Train_Dataset.zip
├─src
|   ├─python             # functions and classes written by python
|   └─matlab             # functions and classes written by matlab
├─requirements.txt       # python third party requirements
├─README.md              # this file
└─codes.py/mat           # other main step code
```

## How to reproduce our result?

## How to use this code?

## Citation
```
@misc{
    title  = {spread_through_air_spaces_segmentation},
    author = {Kuok-Tong Ng, Jia-Wei Liao, Yi-Cheng Hung},
    url    = {},
    year   = {2022}
}
```
