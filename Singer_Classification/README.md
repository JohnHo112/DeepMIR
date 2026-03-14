# HW1: Singer Classification

## Environment Setup
* `Python:  3.10`
* `cuda: 12.2`

Run the following command to install all required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
Download and unzip the HW1 dataset (artist20) in the project root directory:
```bash
unzip hw1.zip
```

## Inference
The test predictions will be saved in `/task2/results/` as:
* `r13942143.json`

Run testing with:
```bash
python task2/test.py --test_data_root <test_data_root> --checkpoint_path <checkpoint_path>
```
* `--test_data_root` -> testing data dir
* `--checkpoint_path` -> model checkpoint

## Task1: Train a Traditional Machine Learning Model
### Preprocessing
This step will:
* Segment songs into 15-second clips
* Extract Mel-spectrogram and MFCC features
* Save the processed data under the `/dataset` directory

The `/dataset` directory contains two subfolders:
* `/train` -> training data
* `/val` -> validation data
* A `.json` file with detailed metadata is also provided.

Run the following command to preprocess the data (without source separation):
```bash
python preprocessing.py
```

### Training & Validation
After training/validation, results will be saved under `/task1/results`, including:
* `confusion_matrix.png` -> confusion matrix
* `val_pred.json` -> validation predictions
* `val_ans.json` -> ground-truth validation labels

Run validation with:
```bash
python task1/valid.py
```

To calculate Top-1 and Top-3 accuracy using the script provided by the TA:
```bash
python count_score.py task1/results/val_ans.json task1/results/val_pred.json
```

## Task 2: Train a Deep Learning Model
### Preprocessing
This step will:
* Segment songs into 15-second clips
* Apply source separation to extract vocals
* Extract Mel-spectrogram and MFCC features
* Save the processed data under `/dataset`

Run the following command for preprocessing with source separation:
```bash
python preprocessing.py --source_sep True
```

### Training
During training, results will be saved under `/task2/results`, including:

* `model.pth` ->Model checkpoints
* `loss.png` -> training loss curve
* `acc.png` -> training accuracy curve

Run training with:
```bash
python task2/train.py
```

### Validation
Validation results are saved under `/task2/results`, including:
* `confusion_matrix.png` -> confusion matrix
* `val_pred.json` -> validation predictions
* `val_ans.json` -> ground-truth validation labels

Run validation with a specific checkpoint:
```bash
python task2/valid.py --checkpoint_path <checkpoint_path>
```

To calculate Top-1 and Top-3 accuracy:
```bash
python count_score.py task2/results/val_ans.json task2/results/val_pred.json
```
