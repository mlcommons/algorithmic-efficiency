#!/usr/bin/env fish

# Benchmark script to compare JAX vs PyTorch ResNet50 model performance
# Usage: ./benchmark_models.fish

set script_dir (dirname (status filename))
set pytorch_output "$script_dir/benchmark_model_pytorch.txt"
set jax_output "$script_dir/benchmark_model_jax.txt"

echo "============================================="
echo "ResNet50 Model Benchmark (Forward + Backward)"
echo "============================================="
echo ""

# Run PyTorch benchmark with DDP (4 processes)
echo ">>> Running PyTorch Model Benchmark (DDP with 4 GPUs)..."
echo ">>> Activating conda environment: ap11_torch_latest"
conda activate ap11_torch_latest

echo ">>> Output will be saved to: $pytorch_output"
torchrun --nproc_per_node=4 --standalone benchmark_model_pytorch.py 2>&1 | tee $pytorch_output
set pytorch_status $status

if test $pytorch_status -ne 0
    echo "PyTorch benchmark failed with status $pytorch_status"
end

echo ""

# Run JAX benchmark
echo ">>> Running JAX Model Benchmark..."
echo ">>> Activating conda environment: ap11_jax"
conda activate ap11_jax

echo ">>> Output will be saved to: $jax_output"
python benchmark_model_jax.py 2>&1 | tee $jax_output
set jax_status $status

if test $jax_status -ne 0
    echo "JAX benchmark failed with status $jax_status"
end

echo ""

# Extract results from output files
function extract_result
    set file $argv[1]
    set key $argv[2]
    grep "^$key=" $file | sed "s/$key=//"
end

# Parse PyTorch results
set pt_forward_ms (extract_result $pytorch_output "FORWARD_MEAN_MS")
set pt_forward_tp (extract_result $pytorch_output "FORWARD_THROUGHPUT")
set pt_train_ms (extract_result $pytorch_output "TRAIN_MEAN_MS")
set pt_train_tp (extract_result $pytorch_output "TRAIN_THROUGHPUT")

# Parse JAX results
set jax_forward_ms (extract_result $jax_output "FORWARD_MEAN_MS")
set jax_forward_tp (extract_result $jax_output "FORWARD_THROUGHPUT")
set jax_train_ms (extract_result $jax_output "TRAIN_MEAN_MS")
set jax_train_tp (extract_result $jax_output "TRAIN_THROUGHPUT")

echo "============================================="
echo "                RESULTS TABLE"
echo "============================================="
echo ""
printf "%-25s %15s %15s\n" "" "PyTorch" "JAX"
echo "-------------------------------------------------------------"
printf "%-25s %12s ms %12s ms\n" "Forward Mean" "$pt_forward_ms" "$jax_forward_ms"
printf "%-25s %12s/s %12s/s\n" "Forward Throughput" "$pt_forward_tp" "$jax_forward_tp"
echo "-------------------------------------------------------------"
printf "%-25s %12s ms %12s ms\n" "Train (Fwd+Bwd) Mean" "$pt_train_ms" "$jax_train_ms"
printf "%-25s %12s/s %12s/s\n" "Train Throughput" "$pt_train_tp" "$jax_train_tp"
echo "-------------------------------------------------------------"
echo ""
echo "Note: PyTorch uses torch.compile, JAX uses jax.jit"
echo "      Both use batch size 1024 (global)"
echo ""
