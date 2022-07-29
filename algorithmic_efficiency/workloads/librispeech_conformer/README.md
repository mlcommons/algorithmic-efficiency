## Installing python dependencies

We use pip to manage our dependencies, `librosa` is the main package needed to
load audio for our preprocessing script.

> pip3 install -r requirements.txt

## Download Raw Data

> chmod a+x download_data.sh
> mkdir data_dir
> cd data_dir
> ../download_data.sh

## Training SPM Tokenizer
this step trains a simple sentence piece tokenizer over librispeech training data.
this tokenizer is then used in later preprocessing step to tokenize transcripts.

> train_tokenizer.py --train=True --data_dir=data_dir/LibriSpeech

the trained tokenizer can be loaded back to do sanity check by tokenizing + de-tokenizing a constant string.

> train_tokenizer.py --train=False

this command will generate `spm_model.vocab` file in the folder it's run

## Run Preprocessing Script

> mkdir work_dir
> cd work_dir
> python3 ../prepare_data.py --data_dir=../data_dir/LibriSpeech --tokenizer_vocab_path=../spm_model.vocab 

the preprocessing script will generate `.npy` files for audio data, `features.csv` which have path to saved audio `.npy`
and `trans.csv` which has path to features.csv and transcription data, individual data loaders 

