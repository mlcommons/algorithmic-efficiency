# Target Setting Run replications
Original runs were run on Google TPUv2-8 machines.

These are not valid submissions, because they use a different hyperparameter setting per workload. But we include them in order to reproduce how we set the target metric values.

To simplify directory setting, set:
```bash
ROOT_DIR=/home/znado
```

## Criteo
Target was set using AdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=criteo1tb \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/criteo1tb/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=criteo1tb \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/criteo1tb/tuning_search_space.json
```

## FastMRI
Target was set using NAdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=fastmri \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_nesterov.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/fastmri/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=fastmri \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_nesterov.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/fastmri/tuning_search_space.json
```

## ImageNet-Resnet
Target was set using Nesterov with a linear warmup and linear decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR \
    --imagenet_v2_data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=imagenet_resnet \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_momentum.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/imagenet_resnet/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR/imagenet_pytorch \
    --imagenet_v2_data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=imagenet_resnet \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_momentum.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/imagenet_resnet/tuning_search_space.json
```

## ImageNet-ViT
Target was set using NAdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR \
    --imagenet_v2_data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=imagenet_vit \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/imagenet_vit/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR/imagenet_pytorch \
    --imagenet_v2_data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=imagenet_vit \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/imagenet_vit/tuning_search_space.json
```

## Librispeech-Conformer
Target was set using AdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=librispeech_conformer \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/librispeech_conformer/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=librispeech_conformer \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/librispeech_conformer/tuning_search_space.json
```

## Librispeech-Deepspeech
Target was set using NAdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=librispeech_deepspeech \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/librispeech_deepspeech/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=librispeech_deepspeech \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/librispeech_deepspeech/tuning_search_space.json
```

## OGBG
Target was set using Nesterov with a linear warmup and linear decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR/tensorflow_datasets \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=ogbg \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_nesterov.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/ogbg/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR/tensorflow_datasets \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=ogbg \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_nesterov.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/ogbg/tuning_search_space.json
```

## WMT
Target was set using AdamW with a linear warmup cosine decay LR schedule.
```bash
python3 submission_runner.py \
    --framework=jax \
    --data_dir=$ROOT_DIR/tensorflow_datasets \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=wmt \
    --submission_path=reference_algorithms/target_setting_algorithms/jax_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/wmt/tuning_search_space.json
```
```bash
torchrun --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 --standalone --nnodes=1 --nproc_per_node=8 submission_runner.py \
    --framework=pytorch \
    --data_dir=$ROOT_DIR/tensorflow_datasets \
    --experiment_dir=$ROOT_DIR \
    --experiment_name=target_setting \
    --workload=wmt \
    --submission_path=reference_algorithms/target_setting_algorithms/pytorch_nadamw.py \
    --tuning_search_space=reference_algorithms/target_setting_algorithms/wmt/tuning_search_space.json
```
