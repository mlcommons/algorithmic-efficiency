# Dataset Setup
TL;DR: 
Use `dataset_setup.py` to download datasets.
Usage:
```bash
python3 datasets/dataset_setup.py \
  --data_dir=~/data \
  --<dataset_name>
  --<optional_fags>
```
The complete benchmark uses 6 datasets:
- OGBG
- WMT
- FastMRI
- Imagenet 
- Criteo 1TB
- Librispeech


Some dataset setups will require you to sign a third party agreement with the 
dataset in order to get the donwload URLs.


# Per dataset instructions
## Environment

### Set data directory (Docker container)
If you are running the `dataset_setup.py` script from a Docker container, please 
make sure the data directory is mounted to a directory on your host with
-v flag. If you are following instructions from the README you will have used 
the `-v $HOME/data:/data` flag in the `docker run` command. This will mount
the `$HOME/data` directory to the `/data` directory in the container. 
In this case set --data_dir to  `\data`. 
```bash
DATA_DIR='/data'
```
### Set data directory (on host)
Alternatively, if you are running the data download script directly on your host, feel free
to choose whatever directory you find suitable, further submission instructions 
assume the data is stored in `~/data`.
```bash
DATA_DIR='~/data'
```
#### Start tmux session (Recommended)
If running the dataset_setup.py on directly on host it is recommended to run 
the dataset_setup.py script in a tmux session because some of the data downloads may 
take several hours. To avoid your setup being interrupted start a tmux session:
```bash
tmux new -s data_setup
```


## Datasets

### OGBG 
From `algorithmic-efficiency` run:
```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR/ogbg \
--ogbg
```

### WMT 
From `algorithmic-efficiency` run:
```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--wmt
```


## FastMRI
Fill out form on https://fastmri.med.nyu.edu/. After filling out the form 
you should get an email containing the URLS for "knee_singlecoil_train",
"knee_singlecoil_val" and "knee_singlecoil_test".  

```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--fastmri \
--fastmri_knee_singlecoil_train_url '<knee_singlecoil_train_url>' \
--fastmri_knee_singlecoil_val_url '<knee_singlecoil_val_url>' \
--fastmri_knee_singlecoil_test_url '<knee_singlecoil_test_url>'
```

## ImageNet
Register on https://image-net.org/ and follow directions to obtain the 
URLS for the ILSVRC2012 train and validation images.

Imagenet dataset processsing is resource intensive. To avoid potential
ResourcExhausted errors increase the maximum number of open file descriptors:
```bash
ulimit -n 8192
```

The imagenet data pipeline differs between the pytorch and jax workloads. 
Therefore, you will have to specify the framework (pytorch or jax) through the
framework flag.

```bash
python3 datasets/dataset_setup.py \ 
--data_dir=/data \
--imagenet \
--temp_dir=$DATA_DIR/tmp \  
--imagenet_train_url=<imagenet_train_url> \
--imagenet_val_url=<imagenet_val_url\
--framework=jax

```

Note that some functions use subprocess.Popen(..., shell=True), which can be
dangerous if the user injects code into the --data_dir or --temp_dir flags. We
do some basic sanitization in main(), but submitters should not let untrusted
users run this script on their systems.

### Cleanup 
In order to avoid potential accidental deletion, this script does NOT
delete any intermediate temporary files (such as zip archives) without a user
confirmation. Deleting temp files is particularly important for Criteo 1TB, as
there can be multiple copies of the dataset on disk during preprocessing if
files are not cleaned up. If you do not want any temp files to be deleted, you
can pass --interactive_deletion=false and then all files will be downloaded to
the provided --temp_dir, and the user can manually delete these after
downloading has finished.

## Criteo1tb
```bash
python3 datasets/dataset_setup.py \
  --data_dir $DATA_DIR \
  --temp_dir $DATA_DIR/tmp \
  --criteo1tb \
```

### Clean up 
In order to avoid potential accidental deletion, this script does NOT
delete any intermediate temporary files (such as zip archives) without a user
confirmation. Deleting temp files is particularly important for Criteo 1TB, as
there can be multiple copies of the dataset on disk during preprocessing if
files are not cleaned up. If you do not want any temp files to be deleted, you
can pass --interactive_deletion=false and then all files will be downloaded to
the provided --temp_dir, and the user can manually delete these after
downloading has finished.


## Librispeech

### Training SPM Tokenizer
This step trains a simple sentence piece tokenizer over librispeech training data.
This tokenizer is then used in later preprocessing step to tokenize transcripts.
This command will generate `spm_model.vocab` file in `$DATA_DIR/librispeech`:
```bash
python3 librispeech_tokenizer.py --train --data_dir=$DATA_DIR/librispeech
```

The trained tokenizer can be loaded back to do sanity check by tokenizing + de-tokenizing a constant string:
```bash
librispeech_tokenizer.py --data_dir=$DATA_DIR/librispeech
```

### Preprocessing Script
The preprocessing script will generate `.npy` files for audio data, `features.csv` which has paths to saved audio `.npy`, and `trans.csv` which has paths to `features.csv` and transcription data.

```bash
python3 librispeech_preprocess.py --data_dir=$DATA_DIR/librispeech --tokenizer_vocab_path=$DATA_DIR/librispeech/spm_model.vocab
```



