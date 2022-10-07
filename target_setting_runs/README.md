# Target Setting Run replications
Original runs were run on Google TPUv2-8 machines.

## Criteo
Target was set using AdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=criteo1tb \
    --submission_path=target_setting_runs/jax_adamw.py \
    --tuning_search_space=target_setting_runs/criteo1tb/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=criteo1tb \
    --submission_path=target_setting_runs/pytorch_adamw.py \
    --tuning_search_space=target_setting_runs/criteo1tb/tuning_search_space.json
```

# FastMRI
Target was set using NAdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=fastmri \
    --submission_path=target_setting_runs/jax_nadamw.py \
    --tuning_search_space=target_setting_runs/fastmri/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=fastmri \
    --submission_path=target_setting_runs/pytorch_nadamw.py \
    --tuning_search_space=target_setting_runs/fastmri/tuning_search_space.json
```

# ImageNet-Resnet
Target was set using Nesterov with a linear warmup and linear decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=imagenet_resnet \
    --submission_path=target_setting_runs/jax_nesterov.py \
    --tuning_search_space=target_setting_runs/imagenet_resnet/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=imagenet_resnet \
    --submission_path=target_setting_runs/pytorch_nesterov.py \
    --tuning_search_space=target_setting_runs/imagenet_resnet/tuning_search_space.json
```

# ImageNet-ViT
Target was set using NAdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=imagenet_vit \
    --submission_path=target_setting_runs/jax_nadamw.py \
    --tuning_search_space=target_setting_runs/imagenet_vit/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=imagenet_vit \
    --submission_path=target_setting_runs/pytorch_nadamw.py \
    --tuning_search_space=target_setting_runs/imagenet_vit/tuning_search_space.json
```

# Librispeech-Conformer
Target was set using AdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=librispeech_conformer \
    --submission_path=target_setting_runs/jax_adamw.py \
    --tuning_search_space=target_setting_runs/librispeech_conformer/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=librispeech_conformer \
    --submission_path=target_setting_runs/pytorch_adamw.py \
    --tuning_search_space=target_setting_runs/librispeech_conformer/tuning_search_space.json
```

# Librispeech-Deepspeech
Target was set using NAdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=librispeech_deepspeech \
    --submission_path=target_setting_runs/jax_nadamw.py \
    --tuning_search_space=target_setting_runs/librispeech_deepspeech/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=librispeech_deepspeech \
    --submission_path=target_setting_runs/pytorch_nadamw.py \
    --tuning_search_space=target_setting_runs/librispeech_deepspeech/tuning_search_space.json
```

# OGBG
Target was set using Nesterov with a linear warmup and linear decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=ogbg \
    --submission_path=target_setting_runs/jax_nesterov.py \
    --tuning_search_space=target_setting_runs/ogbg/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=ogbg \
    --submission_path=target_setting_runs/pytorch_nesterov.py \
    --tuning_search_space=target_setting_runs/ogbg/tuning_search_space.json
```

# WMT
Target was set using AdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --workload=wmt \
    --submission_path=target_setting_runs/jax_adamw.py \
    --tuning_search_space=target_setting_runs/wmt/tuning_search_space.json
```
```bash
python3 submission_runner.py \
    --framework=pytorch \
    --workload=wmt \
    --submission_path=target_setting_runs/pytorch_adamw.py \
    --tuning_search_space=target_setting_runs/wmt/tuning_search_space.json
```
