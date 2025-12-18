#!/usr/bin/env fish

# Benchmark script to compare JAX vs PyTorch ImageNet dataloaders
# Usage: ./benchmark_dataloaders.fish

set script_dir (dirname (status filename))
set pytorch_output "$script_dir/benchmark_dataloader_pytorch.txt"
set jax_output "$script_dir/benchmark_dataloader_jax.txt"

echo "============================================="
echo "ImageNet DataLoader Benchmark"
echo "============================================="
echo ""

# Run PyTorch benchmark with DDP (4 processes)
echo ">>> Running PyTorch DataLoader Benchmark (DDP with 4 GPUs)..."
echo ">>> Activating conda environment: ap11_torch_latest"
conda activate ap11_torch_latest

echo ">>> Output will be saved to: $pytorch_output"
torchrun --nproc_per_node=4 --standalone benchmark_dataloader_pytorch.py 2>&1 | tee $pytorch_output
set pytorch_status $status

if test $pytorch_status -ne 0
    echo "PyTorch benchmark failed with status $pytorch_status"
end

echo ""

# Run JAX benchmark
echo ">>> Running JAX DataLoader Benchmark..."
echo ">>> Activating conda environment: ap11_jax"
conda activate ap11_jax

echo ">>> Output will be saved to: $jax_output"
python benchmark_dataloader_jax.py 2>&1 | tee $jax_output
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
set pt_mean_ms (extract_result $pytorch_output "MEAN_MS")
set pt_throughput (extract_result $pytorch_output "THROUGHPUT")

# Parse JAX results
set jax_mean_ms (extract_result $jax_output "MEAN_MS")
set jax_throughput (extract_result $jax_output "THROUGHPUT")

echo "============================================="
echo "                RESULTS TABLE"
echo "============================================="
echo ""
printf "%-25s %15s %15s\n" "" "PyTorch" "JAX"
echo "-------------------------------------------------------------"
printf "%-25s %12s ms %12s ms\n" "Mean Time per Batch" "$pt_mean_ms" "$jax_mean_ms"
printf "%-25s %12s/s %12s/s\n" "Throughput" "$pt_throughput" "$jax_throughput"
echo "-------------------------------------------------------------"
echo ""
echo "Note: Both use shared TFDS/TFRecords input pipeline"
echo "      Batch size: 1024 (global)"
echo ""
