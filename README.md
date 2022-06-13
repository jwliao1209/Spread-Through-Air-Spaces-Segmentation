# STAS_Final
## Description
This is a competition from [AI CUP 2022](https://tbrain.trendmicro.com.tw/Competitions/Details/22). The goal of this competition is to segment spread through air spaces (STAS) from a lung adenocarcinoma H&E image.

## Requirements
This repository is implemented by `MATLAB2021a` and `python3.8.0` with third party package that stored in `requirements.txt`.

## Dataset and Folder structure
The dataset can be downloaded from [here](https://tbrain.trendmicro.com.tw/Competitions/Details/22). </br>
* For **reproduce our submission result only**, you may just download `Private_Image.zip` and `Public_Image.zip`.
* For **training scheme**, you may need to download `SEG_Train_Dataset.zip` also.

For our training weight files, you may download the weight files **a folder called `checkpoint`** from [here](https://drive.google.com/file/d/1Id2G3Wu3uUeYZV6iBCOrfCYd7475Y51f/view?usp=sharing).

For convinence, you can follow the folder structure as follows
```
.
├─Dataset
|   ├─Private_Image      # From Private_Image.zip
|   ├─Public_Image       # From Private_Image.zip
|   ├─Train_Annotations  # From SEG_Train_Dataset.zip
|   └─Train_Images       # From SEG_Train_Dataset.zip
├─Json_Dataset           # Some .json file for our data pipeline
|   ├─Fold{i}_train.json
|   ├─Fold{i}_valid.json
|   ├─Private.json
|   ├─Public.json
|   └─Test.json
├─checkpoint             # put training weights here
|   ├─0531_4fold
|   ├─0601_4fold
|   └─mix_1
├─configs                # The config files that we train the model
|   ├─Fold0.yaml
|   ├─Fold1.yaml
|   ├─Fold2.yaml
|   └─Fold3.yaml
├─src
|   ├─python             # functions and classes written by python
|   └─matlab             # functions and classes written by matlab
├─requirements.txt       # python third party requirements
├─README.md              # this file
└─codes.py/mat           # other main step code
```

## How to reproduce our result?
To reproduce our result, make sure that you have downloaded the **Public data**, **Private data** and **checkpoint (model weights)**, then place them into the correct location, as described above.
Then, execute `STEP7_Inference.py` and `STEP8_PostProcess.py`, which we are going to explain below.

## How to use this code?
There are total **9 steps** in our procedure (**4 preprocess scripts**, **2 training scripts** and **3 inference scripts**).
### Preprocessing
#### STEP0: Convert `train_annotation.json` to `label.png`
First, we convert the **annotation file (polygon, from `Train_Annotations`)**  to **label (png file)** by using the following command.
```
matlab21b < STEP0_Anno2Mask.m
```
You may find the png folder in `Dataset/Train_Labels`.

#### STEP1: Some simple statistics (optional)
We may observe some simple statistics by running
```
matlab21b < STEP1_Statistics.m
```

#### STEP2: Split training dataset
We handle the data IO by `.json` files (handling by filename). </br>
In this step, we split the training dataset into `Train` and `Test` set.
* `Test`: Testing set, keep fixed.
* `Train`: Training set, split it into train/validation by five fold manner.
Run the following command:
```
matlab21b < STEP2_SplitFiveFold.m
```
You may see those `.json` file in `JsonDataset`. </br>
**Note: We have provided an example in `JsonDataset`, you may skip this step if you want.**

#### STEP3: Make augmentation images
As we state in our report, we have applied **H & E staining extract** for the training dataset. Decomposed an image into three parts: **norm**, **H** and **E**. This can be done by running
```
matlab21b < STEP3_MakeAugment.m
```
You may find the resulting image folders in `Dataset/Train_Eimg', `Dataset/Train_Himg` and `Dataset/Train_norm`.

### Training
#### STEP4: Training
There are two ways to run training script.
* Modify `Config.py` and run `python3 STEP4_Train.py`
* Prepare a config yaml file (e.g, `configs/Fold0.yaml` which we have provide), and run `python3 STEP4_Train.py --config Fold0.yaml`

After that, a checkpoint folder named by current date time (e.g. `2022-06-13-10-27-04`) will be generated in `checkpoint/2022-06-13-10-27-04`, the usage of this folder is to *store trained model weights, which can be catched during testing and inference*.

#### STEP5: Testing
Execute testing scheme by running the following command, note that you should provide a `checkpoint_folder` that we have mentioned in `STEP4`.
```
python3 STEP5_Test.py --checkpont {checkpoint_folder}
```
You could find the testing result inside the `checkpoint_folder`.

### Inference
#### STEP6: Generate Public and Private json files.
As we mentioned in `STEP2`, we need to generate a `.json` file to control data IO. We deal with it by running
```
matlab21b < STEP6_GenerateSubmitJson.m
```
You can find the `.json` files in `JsonDataset`. </br>
**Notice: We have provied an example in `JsonDataset`, If there is no change for the Public/Private dataset, you can skip this step**.

#### STEP7: Inference by model
In order to implement model ensemble in our task, we maintain a checkpoint file so that we don't need to input a bunch of checkpoint folders. </br>
You may modify it in `Checkpoint_setting.py`. After that, we may run inference step by
```
python3 STEP7_Inference.py
```
You may find the inferece result in `prediction/public` and `prediction/private`.

#### STEP8: Post process
For post processing, ensure that you have done `STEP7` and `prediction/public, prediction/private` are exists. Then, we may run the following command
```
matlab21b < STEP8_PostProcess.m
```
You may find the post process result in `prediciton/public-post` and `prediction/private-post`.

## Result
<table>
  <tr>
    <td>Public Score</td>
    <td>Public Rank</td>
    <td>Private Score</td>
    <td>Private Rank</td>
  </tr>
  <tr>
    <td>0.920027</td>
    <td>3</td>
    <td>0.910871</td>
    <td>16</td>
  </tr>
<table>

## Citation
```
@misc{
    title  = {spread_through_air_spaces_segmentation},
    author = {Kuok-Tong Ng, Jia-Wei Liao, Yi-Cheng Hung},
    url    = {https://github.com/K-T-Ng/STAS_Final},
    year   = {2022}
}
```
