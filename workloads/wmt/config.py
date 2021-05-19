"""Default Hyperparameter configuration."""

import types

config = types.SimpleNamespace(
    vocab_path='./wmt_256/sentencepiece_model',
    vocab_size=32000,
    max_corpus_chars=10**7,
    dataset_name='wmt17_translate/de-en',
    eval_split='test',
    reverse_translation=True,
    beam_size=4,
    num_eval_steps=20,
    num_predict_steps=-1,
    learning_rate=0.0625,
    warmup_steps=1000,
    label_smoothing=0.1,
    weight_decay=0.0,
    max_target_length=256,
    max_eval_target_length=256,
    max_predict_length=256,
    share_embeddings=True,
    logits_via_embedding=True,
    num_layers=6,
    qkv_dim=1024,
    emb_dim=1024,
    mlp_dim=4096,
    num_heads=16,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    use_bfloat16=True,
    workdir='./wmt_256',
    per_device_batch_size=64,
    eval_dataset_name='')

