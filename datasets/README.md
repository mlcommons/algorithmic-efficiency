# MLCommons™ AlgoPerf: Dataset Setup

## Table of Contents <!-- omit from toc -->

- [General Setup](#general-setup)
  - [Set Data Directory (Docker Container)](#set-data-directory-docker-container)
  - [Set Data Directory (on Host)](#set-data-directory-on-host)
    - [Start tmux session (Recommended)](#start-tmux-session-recommended)
  - [Clean up](#clean-up)
- [Individual Dataset Instructions](#individual-dataset-instructions)
  - [OGBG](#ogbg)
  - [WMT](#wmt)
  - [FastMRI](#fastmri)
  - [ImageNet](#imagenet)
  - [Criteo1TB](#criteo1tb)
  - [LibriSpeech](#librispeech)
    - [Training SPM Tokenizer](#training-spm-tokenizer)
    - [Preprocessing Script](#preprocessing-script)

## General Setup

This document provides instructions on downloading and preparing all datasets utilized in the AlgoPerf benchmark. You can prepare the individual datasets one-by-one as needed. If your setup, such as your cloud or cluster environment, already contains these datasets, you may skip the dataset setup for this particular data (and directly specify the dataset location in the `submission_runner.py`). Just verify that you are using the same dataset version (and possible preprocessing).

*TL;DR to download and prepare a dataset, run `dataset_setup.py`:*

```bash
python3 datasets/dataset_setup.py \
  --data_dir=~/data \
  --<dataset_name>
  --<optional_flags>
```

The complete benchmark uses 6 different datasets:

- [OGBG](#ogbg)
- [WMT](#wmt)
- [FastMRI](#fastmri)
- [Imagenet](#imagenet)
- [Criteo 1TB](#criteo1tb)
- [Librispeech](#librispeech)

Some dataset setups will require you to sign a third-party agreement with the dataset owners in order to get the download URLs.

### Set Data Directory (Docker Container)

If you are running the `dataset_setup.py` script from a Docker container, please
make sure the data directory is mounted to a directory on your host with
`-v` flag. If you are following instructions from the [Getting Started guide](/GETTING_STARTED.md) you will have used
the `-v $HOME/data:/data` flag in the `docker run` command. This will mount
the `$HOME/data` directory to the `/data` directory in the container.
In this case set, `--data_dir` to  `/data`.

```bash
DATA_DIR='/data'
```

### Set Data Directory (on Host)

Alternatively, if you are running the data download script directly on your host, feel free to choose whatever directory you find suitable, further submission instructions assume the data is stored in `~/data`.

```bash
DATA_DIR='~/data'
```

#### Start tmux session (Recommended)

If running the `dataset_setup.py` on directly on host it is recommended to run
the `dataset_setup.py` script in a `tmux` session because some of the data downloads may take several hours. To avoid your setup being interrupted start a `tmux` session:

```bash
tmux new -s data_setup
```

### Clean up

In order to avoid potential accidental deletion, this script does NOT
delete any intermediate temporary files (such as zip archives) without a user
confirmation. Deleting temp files is particularly important for Criteo 1TB, as
there can be multiple copies of the dataset on disk during preprocessing if
files are not cleaned up.

By default, a user will be prompted before any files are deleted. If you do not want any temp files to be deleted, you can pass `--interactive_deletion=false` and then all files will be downloaded to the provided `--temp_dir`, and the user can manually delete these after downloading has finished.

## Individual Dataset Instructions

### OGBG

From `algorithmic-efficiency` run:

```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--ogbg
```

<details>
<summary>The final directory structure should look like this:</summary>

```bash
$DATA_DIR
├── ogbg
│   └── ogbg_molpcba
│       └── 0.1.3
│           ├── dataset_info.json
│           ├── features.json
│           ├── metadata.json
│           ├── ogbg_molpcba-test.tfrecord-00000-of-00001
│           ├── ogbg_molpcba-train.tfrecord-00000-of-00008
│           ├── ogbg_molpcba-train.tfrecord-00001-of-00008
│           ├── ogbg_molpcba-train.tfrecord-00002-of-00008
│           ├── ogbg_molpcba-train.tfrecord-00003-of-00008
│           ├── ogbg_molpcba-train.tfrecord-00004-of-00008
│           ├── ogbg_molpcba-train.tfrecord-00005-of-00008
│           ├── ogbg_molpcba-train.tfrecord-00006-of-00008
│           ├── ogbg_molpcba-train.tfrecord-00007-of-00008
│           └── ogbg_molpcba-validation.tfrecord-00000-of-00001
```

In total, it should contain 13 files (via `find -type f | wc -l`) for a total of 777 MB (via `du -sch ogbg/`).
</details>

### WMT

From `algorithmic-efficiency` run:

```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--wmt
```

<details>
<summary>The final directory structure should look like this:</summary>

```bash
$DATA_DIR
├── wmt
    ├── wmt14_translate
    │   └── de-en
    │       └── 1.0.0
    │           ├── dataset_info.json
    │           ├── features.json
    │           ├── wmt14_translate-test.tfrecord-00000-of-00001
    │           ├── wmt14_translate-train.tfrecord-00000-of-00016
    │           ├── wmt14_translate-train.tfrecord-00001-of-00016
    │           ├── wmt14_translate-train.tfrecord-00002-of-00016
    │           ├── wmt14_translate-train.tfrecord-00003-of-00016
    │           ├── wmt14_translate-train.tfrecord-00004-of-00016
    │           ├── wmt14_translate-train.tfrecord-00005-of-00016
    │           ├── wmt14_translate-train.tfrecord-00006-of-00016
    │           ├── wmt14_translate-train.tfrecord-00007-of-00016
    │           ├── wmt14_translate-train.tfrecord-00008-of-00016
    │           ├── wmt14_translate-train.tfrecord-00009-of-00016
    │           ├── wmt14_translate-train.tfrecord-00010-of-00016
    │           ├── wmt14_translate-train.tfrecord-00011-of-00016
    │           ├── wmt14_translate-train.tfrecord-00012-of-00016
    │           ├── wmt14_translate-train.tfrecord-00013-of-00016
    │           ├── wmt14_translate-train.tfrecord-00014-of-00016
    │           ├── wmt14_translate-train.tfrecord-00015-of-00016
    │           └── wmt14_translate-validation.tfrecord-00000-of-00001
    ├── wmt17_translate
    │   └── de-en
    │       └── 1.0.0
    │           ├── dataset_info.json
    │           ├── features.json
    │           ├── wmt17_translate-test.tfrecord-00000-of-00001
    │           ├── wmt17_translate-train.tfrecord-00000-of-00016
    │           ├── wmt17_translate-train.tfrecord-00001-of-00016
    │           ├── wmt17_translate-train.tfrecord-00002-of-00016
    │           ├── wmt17_translate-train.tfrecord-00003-of-00016
    │           ├── wmt17_translate-train.tfrecord-00004-of-00016
    │           ├── wmt17_translate-train.tfrecord-00005-of-00016
    │           ├── wmt17_translate-train.tfrecord-00006-of-00016
    │           ├── wmt17_translate-train.tfrecord-00007-of-00016
    │           ├── wmt17_translate-train.tfrecord-00008-of-00016
    │           ├── wmt17_translate-train.tfrecord-00009-of-00016
    │           ├── wmt17_translate-train.tfrecord-00010-of-00016
    │           ├── wmt17_translate-train.tfrecord-00011-of-00016
    │           ├── wmt17_translate-train.tfrecord-00012-of-00016
    │           ├── wmt17_translate-train.tfrecord-00013-of-00016
    │           ├── wmt17_translate-train.tfrecord-00014-of-00016
    │           ├── wmt17_translate-train.tfrecord-00015-of-00016
    │           └── wmt17_translate-validation.tfrecord-00000-of-00001
    └── wmt_sentencepiece_model
```

In total, it should contain 43 files (via `find -type f | wc -l`) for a total of 3.3 GB (via `du -sch wmt/`).
</details>

### FastMRI

Fill out form on <https://fastmri.med.nyu.edu/>. After filling out the form
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

<details>
<summary>The final directory structure should look like this:</summary>

```bash
$DATA_DIR
├── fastmri
│   ├── knee_singlecoil_test
│   │   ├── file1000022.h5
│   │   ├── [...]
│   │   └── file1002571.h5
│   ├── knee_singlecoil_train
│   │   ├── file1000001.h5
│   │   ├── [...]
│   │   └── file1002569.h5
│   └── knee_singlecoil_val
│       ├── file1000000.h5
│       ├── [...]
│       └── file1002570.h5
```

In total, it should contain 1280 files (via `find -type f | wc -l`) for a total of 112 GB (via `du -sch fastmri/`).
</details>

### ImageNet

Register on <https://image-net.org/> and follow directions to obtain the
URLS for the ILSVRC2012 train and validation images.
The script will additionally automatically download the `matched-frequency` version of [ImageNet v2](https://www.tensorflow.org/datasets/catalog/imagenet_v2#imagenet_v2matched-frequency_default_config), which is used as the test set of the ImageNet workloads.

The ImageNet data pipeline differs between the PyTorch and JAX workloads.
Therefore, you will have to specify the framework (either `pytorch` or `jax`) through the framework flag.

```bash
python3 datasets/dataset_setup.py \ 
--data_dir $DATA_DIR \
--imagenet \
--temp_dir $DATA_DIR/tmp \  
--imagenet_train_url <imagenet_train_url> \
--imagenet_val_url <imagenet_val_url> \
--framework jax
```

Imagenet dataset processsing is resource intensive. To avoid potential
ResourcExhausted errors increase the maximum number of open file descriptors:

```bash
ulimit -n 8192
```

Note that some functions use `subprocess.Popen(..., shell=True)`, which can be
dangerous if the user injects code into the `--data_dir` or `--temp_dir` flags. We
do some basic sanitization in `main()`, but submitters should not let untrusted
users run this script on their systems.

<details>
<summary>The final directory structure should look like this for ImageNet2012 (PyTorch):</summary>

```bash
$DATA_DIR
├── imagenet
│   ├── train
│       ├── n01440764
│           ├── n01440764_10026.JPEG
│           ├── n01440764_10027.JPEG
│           ├── n01440764_10029.JPEG
│           ├── [...]
│       ├── [...]
│   └── val
│       ├── n01440764
│           ├── ILSVRC2012_val_00000293.JPEG
│           ├── ILSVRC2012_val_00002138.JPEG
│           ├── [...]
│       ├── [...]
```

In total, it should contain 1,281,167 `train` files and 50,000 `val` (via `find -type f | wc -l`) for a total of 177 GB and 7.8 GB, respectively (via `du -sch train/` and `du -sch val/`).
</details>

**TODO**
<details>
<summary>The final directory structure should look like this for ImageNet2012 (JAX):</summary>

```bash
$DATA_DIR
```

In total, it should contain ?? files (via `find -type f | wc -l`) for a total of ?? GB (via `du -sch imagenet/`).
</details>

<details>
<summary>The final directory structure should look like this for ImageNet v2:</summary>

```bash
$DATA_DIR
├── imagenet_v2
│   └── matched-frequency
│       └── 3.0.0
│           ├── dataset_info.json
│           ├── features.json
│           ├── imagenet_v2-test.tfrecord-00000-of-00016
│           ├── imagenet_v2-test.tfrecord-00001-of-00016
│           ├── imagenet_v2-test.tfrecord-00002-of-00016
│           ├── imagenet_v2-test.tfrecord-00003-of-00016
│           ├── imagenet_v2-test.tfrecord-00004-of-00016
│           ├── imagenet_v2-test.tfrecord-00005-of-00016
│           ├── imagenet_v2-test.tfrecord-00006-of-00016
│           ├── imagenet_v2-test.tfrecord-00007-of-00016
│           ├── imagenet_v2-test.tfrecord-00008-of-00016
│           ├── imagenet_v2-test.tfrecord-00009-of-00016
│           ├── imagenet_v2-test.tfrecord-00010-of-00016
│           ├── imagenet_v2-test.tfrecord-00011-of-00016
│           ├── imagenet_v2-test.tfrecord-00012-of-00016
│           ├── imagenet_v2-test.tfrecord-00013-of-00016
│           ├── imagenet_v2-test.tfrecord-00014-of-00016
│           ├── imagenet_v2-test.tfrecord-00015-of-00016
│           └── label.labels.txt
```

In total, it should contain 20 files (via `find -type f | wc -l`) for a total of 1.2 GB (via `du -sch imagenet_v2/`).
</details>

### Criteo1TB

```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--temp_dir $DATA_DIR/tmp \
--criteo1tb 
```

**TODO**
<details>
<summary>The final directory structure should look like this:</summary>

```bash
$DATA_DIR
```

In total, it should contain ?? files (via `find -type f | wc -l`) for a total of ?? GB (via `du -sch criteo1tb/`).
</details>

### LibriSpeech

To download, train a tokenizer and preprocess the librispeech dataset:

```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--temp_dir $DATA_DIR/tmp \
--librispeech
```

**TODO**
<details>
<summary>The final directory structure should look like this:</summary>

```bash
$DATA_DIR
```

In total, it should contain ?? files (via `find -type f | wc -l`) for a total of ?? GB (via `du -sch librispeech/`).
</details>

#### Training SPM Tokenizer

 A simple sentence piece tokenizer is trained over librispeech training
 data. This tokenizer is then used in later preprocessing step to tokenize transcripts.
This command generates `spm_model.vocab` file in `$DATA_DIR/librispeech`:

```bash
python3 librispeech_tokenizer.py --train --data_dir=$DATA_DIR/librispeech
```

The trained tokenizer can be loaded back to do sanity check by tokenizing + de-tokenizing a constant string:

```bash
librispeech_tokenizer.py --data_dir=$DATA_DIR/librispeech
```

#### Preprocessing Script

The preprocessing script will generate `.npy` files for audio data, `features.csv` which has paths to saved audio `.npy`, and `trans.csv` which has paths to `features.csv` and transcription data.

```bash
python3 librispeech_preprocess.py --data_dir=$DATA_DIR/librispeech --tokenizer_vocab_path=$DATA_DIR/librispeech/spm_model.vocab
```
