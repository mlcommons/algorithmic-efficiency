# Dataset Setup
Use `dataset_setup.py` to download datasets.


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
