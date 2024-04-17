# Goznak Test Task 
## Preparations
### Create and activate virtual environment
```bash
conda create -n goznak python==3.11
conda activate goznak
```
### Install requirements
```bash
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```
## Setup "GOZNAK_REPO_DIR" environment variable
### Windows
- Press "Win+S".
- Type "env".
- Select "Edit system environment variables".
- Add user environment variable "GOZNAK_REPO_PATH".
- Set this variable equal to path to this repository.
- Save and exit.

### Linux
- Install **vim**.
```bash
sudo apt install vim
```
- Open .bashrc with vim.
```bash
vim ~/.bashrc
```
- Append the the following (adjust the path to an exact path to repository).
```bash
export GOZNAK_REPO_PATH="path/to/repo"
```
- Apply changes.
```bash
source ~/.bashrc
```

# Run code
For all further sections we will consider, that your CWD (current working directory) is GOZNAK_REPO_PATH. We will also consider, that you have all the requirements installed in 'goznak' conda environment, which will be considered activated.
## Multiplicate
```bash
python src/multiplicate/multiplicate.py
```
## Classify
### Train
```bash
python src/classify/preprocess.py
python src/classify/train.py
```
### Evaluate
```bash
python src/classify/eval.py
```
## Denoise
### Train
```bash
python src/denoise/train.py
```
### Evaluate
```bash
python src/denoise/eval.py
```

For further info, please, check out:
- "data/README.txt" - for more info about training/evaluation data.
- Source code of train/eval (both for classify and denoise) for more details about input arguments.