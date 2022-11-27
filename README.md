# Judge a Book by its Cover
[Kaggle competition](https://www.kaggle.com/competitions/col774-2022/overview)

## Team
- Burouj Armgaan
- Mohit Sharma

## Data
Find `data.zip` at the following [link](https://drive.google.com/file/d/1SyPFq_rb8Cr7ZxcI7H61D6dhusUZGy1L/view?usp=share_link).

## Conda environment
- Run `conda env create -f env.yaml` to create a conda env by the name `perk`.
- Run `conda activate perk` prior to running any scripts in this repo.

## Expedted dataset directory structure
- `<dataset_dir>`
    - `images/`
    - `comp_test_x.csv`
    - `non_comp_test_x.csv`
    - `non_comp_test_y.csv`
    - `train_x.csv`
    - `train_y.csv`

## Run
- CNN : `python cnn.py <dataset_dir>`
- Bidirectional-RNN : `python rnn.py <dataset_dir>`
- Multimodal : `python comp.py <dataset_dir>`

## Description
- Running any of the `.py` files will do the following:
    - Train the model
    - Print out the test accuracy
    - Save the model to disk in the format `<model>_<time>.pt`.
    - Store the predictions on the test set in a csv file named:
        - `non_comp_test_pred_y.csv` for `cnn.py` & `rnn.py`.
        - `comp_test_pred_y.csv` for `compy.py`.