## Unix
```
python3 -m venv .venv
source ./.venv/bin/activate
pip3 install -r requirements.txt
```

## Windows
```
python3 -m venv .venv
.\.venv\Scripts\activate
pip3 install -r requirements.txt
```

# Distribution

# Augmentation

# Transformation

# Train
```bash
python3.10 .\src\Train.py [folder source containing images to train] --dataset-dest [destination directory for train/validation dataset] --augment --transform
```

```bash
usage: Train.py [-h] --dataset-dest DATASET_DEST [--augment] [--transform] [--save-checkpoints] [--batch-size BATCH_SIZE] [--random-seed RANDOM_SEED] [--split SPLIT] src

positional arguments:
  src                   Source directory containing images to train (will be overwritten with transformations)

options:
  -h, --help            show this help message and exit
  --dataset-dest DATASET_DEST
                        Dataset saving directory
  --augment             Agument the image before training.
  --transform           Transform the image before training.
  --save-checkpoints    Save model at each epochs.
  --batch-size BATCH_SIZE
  --random-seed RANDOM_SEED
  --split SPLIT
  ```


# Predict
```bash
python3.10 .\src\Predict.py [directory containing image to test or single file] --rand
```

```bash
usage: Predict.py [-h] [--model MODEL] [--classnames CLASSNAMES] [--rand] path

positional arguments:
  path                  Path to the image or directory to predict.

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to the model.
  --classnames CLASSNAMES
                        Path to the file containing the classnames for the trained model
  --rand                Take random files from the folder
```