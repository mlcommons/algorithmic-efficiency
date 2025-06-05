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

In total, it should contain 13 files (via `find -type f | wc -l`) for a total of 830 MB (via `du -sch --apparent-size ogbg/`).
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

In total, it should contain 43 files (via `find -type f | wc -l`) for a total of 3.3 GB (via `du -sch --apparent-size wmt/`).
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

In total, it should contain 1280 files (via `find -type f | wc -l`) for a total of 113 GB (via `du -sch --apparent-size fastmri/`).
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

Clean up: we do not remove the leftover `.tar` files in preprocessing. If setting
up this dataset for the `pytorch` framework, please remove leftover `.tar`
files `$DATA_DIR/imagenet/pytorch` and `$DATA_DIR/imagenet/pytorch/train`.

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

In total, it should contain 1,281,167 `train` files and 50,000 `val` (via `find -type f | wc -l`) for a total of 137 GB and 6.3 GB, respectively (via `du -sch --apparent-size train/` and `du -sch --apparent-size val/`).
</details>

<details>
<summary>The final directory structure should look like this for ImageNet2012 (JAX) (including v2):</summary>

```bash
$DATA_DIR
├──imagenet
│  ├── jax
│  │   ├── downloads
│  │   │   ├── extracted
│  │   │   └── manual
│  │   ├── imagenet2012
│  │   │   └── 5.1.0
│  │   │       ├── dataset_info.json
│  │   │       ├── features.json
│  │   │       ├── imagenet2012-train.tfrecord-00000-of-01024
│  │   │       ├── imagenet2012-train.tfrecord-00001-of-01024
│  │   │       ├── [...]
│  │   └── imagenet_v2
│  │       └── matched-frequency
│  │           └── 3.0.0
│  │               ├── dataset_info.json
│  │               ├── features.json
│  │               ├── imagenet_v2-test.tfrecord-00000-of-00016
│  │               ├── imagenet_v2-test.tfrecord-00001-of-00016
│  │               ├── [...]
```

In total, it should contain 1,111 files (via `find -type f | wc -l`) for a total of 145 GB (via `du -sch --apparent-size imagenet/jax`).
</details>

<details>
<summary>The final directory structure should look like this for ImageNet v2 (separate):</summary>

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

In total, it should contain 20 files (via `find -type f | wc -l`) for a total of 1.2 GB (via `du -sch --apparent-size imagenet_v2/`).
</details>

### Criteo1TB

```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--temp_dir $DATA_DIR/tmp \
--criteo1tb 
```

Note, that this requries the [`pigz` library](https://zlib.net/pigz/) to be installed.

<details>
<summary>The final directory structure should look like this:</summary>

```bash
$DATA_DIR
├── criteo1tb
│  ├── day_0_00
│  ├── day_0_01
│  ├── day_0_02
│  ├── day_0_03
│  ├── [...]
```

In total, it should contain 885 files (via `find -type f | wc -l`) for a total of 1.1 TB (via `du -sch --apparent-size criteo1tb/`).
</details>

### LibriSpeech

To download, train a tokenizer and preprocess the librispeech dataset:

```bash
python3 datasets/dataset_setup.py \
--data_dir $DATA_DIR \
--temp_dir $DATA_DIR/tmp \
--librispeech
```

Note, that this requries the [`ffmpeg` toolbox](https://ffmpeg.org/) to be installed.

<details>
<summary>The final directory structure should look like this:</summary>

```bash
$DATA_DIR
├──librispeech
│  ├── dev-clean.csv
│  ├── dev-other.csv
│  ├── spm_model.vocab
│  ├── test-clean.csv
│  ├── train-clean-100.csv
│  ├── train-clean-360.csv
│  ├── train-clean-500.csv
│  ├── dev-clean
│  │   ├── 1272-128104-0000_audio.npy
│  │   ├── 1272-128104-0000_targets.npy
│  │   ├── 1272-128104-0001_audio.npy
│  │   ├── 1272-128104-0001_targets.npy
│  │   ├── [...]
│  ├── dev-other
│  │   ├── 116-288045-0000_audio.npy
│  │   ├── 116-288045-0000_targets.npy
│  │   ├── [...]
│  ├── test-clean
│  │   ├── 1089-134686-0000_audio.npy  
│  │   ├── 1089-134686-0000_targets.npy
│  │   ├── [...]
│  ├── train-clean-100
│  │   ├── 103-1240-0000_audio.npy
│  │   ├── 103-1240-0000_targets.npy
│  │   ├── [...]
│  ├── train-clean-360
│  │   ├── 100-121669-0000_audio.npy
│  │   ├── 100-121669-0000_targets.npy
│  │   ├── [...]
│  ├── train-other-500
│  │   ├── 1006-135212-0000_audio.npy
│  │   ├── 1006-135212-0000_targets.npy
│  │   ├── [...]
```

In total, it should contain 543,323 files (via `find -type f | wc -l`) for a total of 387 GB (via `du -sch --apparent-size librispeech/`).
</details>

#### Training SPM Tokenizer

During the above commands, a simple sentence piece tokenizer is trained over librispeech training data.
This tokenizer is then used in later preprocessing step to tokenize transcripts.
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
