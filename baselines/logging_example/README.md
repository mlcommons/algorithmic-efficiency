# Logging Measurements to Disk with `--logging_dir`

## Summary of Measurements Collected
|                                               | `workload_`<br />`results.json` | `trial_`<br />`results.json`  | `eval_`<br />`results.csv` |
|-----------------------------------------------|-----------------------|---------------------|------------------|
| **Basic info** <br /> (ex. datetime)                     | ✔️                    | ✔️                  | ✔️               |
| **Extra metadata** <br /> (optional user-defined)        | ✔️                    | ✔️                  | ✔️               |
| **Workload info** <br /> (ex. target_value)              | ✔️                    | ✔️                  | ✔️               |
| **Workload score**                                | ✔️                    |                     |                  |
| **System software info** <br /> (ex. os_platform)        | ✔️                    |                     |                  |
| **System hardware info** <br /> (ex. gpu_model_name)     | ✔️                    |                     |                  |
| **Tuning search space** <br /> (if used)                 | ✔️                    |                     |                  |
| **Trial hyperparameters**                         |                       | ✔️                  | ✔️               |
| **Hardware   utilization** <br /> (ex. gpu.avg.mem.util) |                       | ✔️                  | ✔️               |
| **Model eval metrics** <br /> (ex. loss)                 |                       | ✔️ (only<br /> last eval) | ✔️ (all evals)   |
| **Stopping   conditions** <br /> (ex. is_time_remaining) |                       | ✔️                  | ✔️               |
| **Step count**                                    |                       | ✔️*                  | ✔️              |
| **Accumulated time**                              |                       | ✔️*                  | ✔️              |

&ast; Caveat: When the tuning trial ran out of time before reaching the target value, one or more steps may have been completed since the last model evaluation. In this case the step count and accumulated time in `trial_results.json` will not exactly reflect when the model evaluation metrics were recorded.

# Usage Tutorial

Summary:
- Choose a directory where you want to save measurements.
- Run a workload with the `--logging_dir` option.
- Configure optional features and plot the results.

__Goal:__ We are going to use the included mnist_jax workload to train an MLP model, record the loss curve, and plot it.

### 1. Create output location

Choose a directory where you want to save measurements:
```bash
$ LOGGING_DIR=./logs
$ mkdir -p $LOGGING_DIR
```

### 2. Run a Workload with with the `--logging_dir` option

Here we run the simplest workload, an MLP JAX model with the MNIST dataset, for 2 training trials with hyperparameters randomly picked from the acceptable range specified in `tuning_search_space.json`.
```bash
$ python3 algorithmic_efficiency/submission_runner.py \
    --framework=jax \
    --workload=mnist_jax \
    --submission_path=baselines/mnist/mnist_jax/submission.py \
    --tuning_search_space=baselines/mnist/tuning_search_space.json \
    --num_tuning_trials=2 \
    --logging_dir=$LOGGING_DIR 2>&1 | tee -a $LOGGING_DIR/console_output.log
```

The saved data is organized in a directory structure like this:
```
./logs/console_output.log
./logs/[workload_name]/workload_results.json
./logs/[workload_name]/packages.txt
./logs/[workload_name]/trial_[n]/trial_results.json
./logs/[workload_name]/trial_[n]/eval_results.csv
```

What are these output files?

Four files are written to the `logging_dir` folder:
  1. `workload_results.json` includes the datetime, workload name, score,
    and the system configuration used to generate the result.
  2. `trial_[n]/trial_results.json` is created at the end of each
    hyperparameter tuning trial and includes the result of the trial. The
    last row of eval_results.csv and this file are very similar but can
    differ in the number of steps and runtime in one situation: when the
    tuning trial ran out of time but completed one or more steps after the
    last model evaluation.
  3. `trial_[n]/eval_results.csv` is created for each hyperparameter tuning
    trial. A row is appended for every model evaluation. The information
    included is loss, accuracy, training step, time elapsed, hparams, workload
    properties, and hardware utilization.
  4. `packages.txt` is created at the start of a workload and it includes a
    list of the currently installed OS and python packages.

### 3. Inspect the output of `workload_results.json`

```
{
    "workload": "mnist",
    "datetime": "2022-04-02T21:56:23.389015",
    "status": "COMPLETE",
    "score": 41.74883270263672,
    "logging_dir": "./logs",
    "submission_path": "baselines/mnist/mnist_jax/submission.py",
    "tuning_ruleset": "external",
    "num_tuning_trials": 2,
    "tuning_search_space_path": "baselines/mnist/tuning_search_space.json",
    "tuning_search_space": {
        "learning_rate": {
            "min": 0.0001,
            "max": 0.01,
            "scaling": "log"
        },
        "one_minus_beta_1": {
            "min": 0.9,
            "max": 0.999,
            "scaling": "log"
        },
        "epsilon": {
            "feasible_points": [
                1e-08,
                1e-05,
                0.001
            ]
        }
    },
    "workload.eval_period_time_sec": 10,
    "workload.max_allowed_runtime_sec": 60,
    "workload.num_eval_train_examples": 10000,
    "workload.num_train_examples": 60000,
    "workload.num_validation_examples": 10000,
    "workload.target_value": 0.9,
    "workload.train_mean": 0.1307,
    "workload.train_stddev": 0.3081,
    "os_platform": "Linux-5.4.48-x86_64-with-glibc2.29",
    "python_version": "3.8.10",
    "python_compiler": "GCC 9.3.0",
    "git_branch": "main",
    "git_commit_hash": "b6ed8b11634b2721c0545e294c69c4986ef47c9f",
    "cpu_model_name": "AMD EPYC 7601 32-Core Processor",
    "cpu_count": 64,
    "gpu_model_name": "GeForce RTX 2080 Ti",
    "gpu_count": 4,
    "gpu_driver": "460.27.04"
}
```

### 4. Inspect the output of `trial_results.json`

```
{
    "workload": "mnist",
    "trial_idx": 1,
    "datetime": "2022-04-02T21:55:31.206899",
    "accumulated_submission_time": 60.00532341003418,
    "accuracy": 0.8566000461578369,
    "loss": 0.4808649718761444,
    "global_step": 1292,
    "epoch": 22.275862068965516,
    "steps_per_epoch": 58,
    "global_start_time": 1648936455.7184513,
    "goal_reached": false,
    "is_time_remaining": false,
    "training_complete": false,
    "batch_size": 1024,
    "hparam.learning_rate": 0.004102837916179185,
    "hparam.one_minus_beta_1": 0.9980795034695658,
    "hparam.epsilon": 0.001,
    "workload.eval_period_time_sec": 10,
    "workload.max_allowed_runtime_sec": 60,
    "workload.num_eval_train_examples": 10000,
    "workload.num_train_examples": 60000,
    "workload.num_validation_examples": 10000,
    "workload.target_value": 0.9,
    "workload.train_mean": 0.1307,
    "workload.train_stddev": 0.3081,
    "cpu.util.avg_percent_since_last": 41.7,
    "cpu.freq.current": 2.6715624999999994,
    "temp.k10temp.current": 52.375,
    "mem.total": 135058636800,
    "mem.available": 73001000960,
    "mem.used": 51393884160,
    "mem.percent_used": 45.9,
    "mem.read_bytes_since_boot": 85397650348032,
    "mem.write_bytes_since_boot": 3221685143552,
    "net.bytes_sent_since_boot": 21568838,
    "net.bytes_recv_since_boot": 679624468,
    "gpu.count": 4,
    "gpu.0.compute.util": 0.96,
    "gpu.0.mem.util": 0.8816589527180325,
    "gpu.0.mem.total": 11019.0,
    "gpu.0.mem.used": 9715.0,
    "gpu.0.mem.free": 1304.0,
    "gpu.0.temp.current": 72.0,
    "gpu.1.compute.util": 0.99,
    "gpu.1.mem.util": 0.8816589527180325,
    "gpu.1.mem.total": 11019.0,
    "gpu.1.mem.used": 9715.0,
    "gpu.1.mem.free": 1304.0,
    "gpu.1.temp.current": 70.0,
    "gpu.2.compute.util": 0.99,
    "gpu.2.mem.util": 0.8883746256466104,
    "gpu.2.mem.total": 11019.0,
    "gpu.2.mem.used": 9789.0,
    "gpu.2.mem.free": 1230.0,
    "gpu.2.temp.current": 73.0,
    "gpu.3.compute.util": 0.98,
    "gpu.3.mem.util": 0.884018513476722,
    "gpu.3.mem.total": 11019.0,
    "gpu.3.mem.used": 9741.0,
    "gpu.3.mem.free": 1278.0,
    "gpu.3.temp.current": 76.0,
    "gpu.avg.compute.util": 0.98,
    "gpu.avg.mem.util": 0.8839277611398493,
    "gpu.avg.mem.total": 11019.0,
    "gpu.avg.mem.used": 9740.0,
    "gpu.avg.mem.free": 1279.0,
    "gpu.avg.temp.current": 72.75,
}
```


### 5. Inspect the output of `eval_results.csv` (only one row shown)

Every module evaluation appends a row to the CSV that looks like:

| workload | trial_idx | datetime                   | accumulated_submission_time | accuracy           | loss               | global_step | epoch | steps_per_epoch | global_start_time  | goal_reached | is_time_remaining | training_complete | batch_size | hparam.learning_rate | hparam.one_minus_beta_1 | hparam.epsilon | workload.eval_period_time_sec | workload.max_allowed_runtime_sec | workload.num_eval_train_examples | workload.num_train_examples | workload.num_validation_examples | workload.target_value | workload.train_mean | workload.train_stddev | cpu.util.avg_percent_since_last | cpu.freq.current | temp.k10temp.current | mem.total    | mem.available | mem.used    | mem.percent_used | mem.read_bytes_since_boot | mem.write_bytes_since_boot | net.bytes_sent_since_boot | net.bytes_recv_since_boot | gpu.count | gpu.0.compute.util | gpu.0.mem.util     | gpu.0.mem.total | gpu.0.mem.used | gpu.0.mem.free | gpu.0.temp.current | gpu.1.compute.util | gpu.1.mem.util     | gpu.1.mem.total | gpu.1.mem.used | gpu.1.mem.free | gpu.1.temp.current | gpu.2.compute.util | gpu.2.mem.util     | gpu.2.mem.total | gpu.2.mem.used | gpu.2.mem.free | gpu.2.temp.current | gpu.3.compute.util | gpu.3.mem.util    | gpu.3.mem.total | gpu.3.mem.used | gpu.3.mem.free | gpu.3.temp.current | gpu.avg.compute.util | gpu.avg.mem.util   | gpu.avg.mem.total | gpu.avg.mem.used | gpu.avg.mem.free | gpu.avg.temp.current |
|----------|-----------|----------------------------|-----------------------------|--------------------|--------------------|-------------|-------|-----------------|--------------------|--------------|-------------------|-------------------|------------|----------------------|-------------------------|----------------|-------------------------------|----------------------------------|----------------------------------|-----------------------------|----------------------------------|-----------------------|---------------------|-----------------------|---------------------------------|------------------|----------------------|--------------|---------------|-------------|------------------|---------------------------|----------------------------|---------------------------|---------------------------|-----------|--------------------|--------------------|-----------------|----------------|----------------|--------------------|--------------------|--------------------|-----------------|----------------|----------------|--------------------|--------------------|--------------------|-----------------|----------------|----------------|--------------------|--------------------|-------------------|-----------------|----------------|----------------|--------------------|----------------------|--------------------|-------------------|------------------|------------------|----------------------|
| mnist    | 1         | 2022-04-02T21:54:26.057695 | 6.426610469818115           | 0.1949000060558319 | 2.2479422092437744 | 0           | 0.0   | 58              | 1648936455.7184513 | False        | True              | False             | 1024       | 0.0041028379161791   | 0.9980795034695658      | 0.001          | 10                            | 60                               | 10000                            | 60000                       | 10000                            | 0.9                   | 0.1307              | 0.3081                | 46.6                            | 2.666234375      | 52.0                 | 135058636800 | 73530703872   | 50940600320 | 45.6             | 85374736922624            | 3221680825344              | 21568838                  | 679624361                 | 4         | 0.95               | 0.8816589527180325 | 11019.0         | 9715.0         | 1304.0         | 71.0               | 0.97               | 0.8816589527180325 | 11019.0         | 9715.0         | 1304.0         | 69.0               | 0.98               | 0.8883746256466104 | 11019.0         | 9789.0         | 1230.0         | 73.0               | 0.96               | 0.884018513476722 | 11019.0         | 9741.0         | 1278.0         | 76.0               | 0.965                | 0.8839277611398493 | 11019.0           | 9740.0           | 1279.0           | 72.25                |


### 6. (Optional) Combine trial JSON files if you want the measurements at the end of all trials in 1 file

You can join all files named `trial_[n]/trial_results.json` in a given folder recursively with this bash command:

```bash
$ jq -s 'flatten' ./logs*/**/trial_**/trial_results.json > all_trials.json
```

### 7. (Optional) Combine eval CSV files if you want all measurements throughout all trials in 1 file

By default, one `eval_results.csv` is produced per training run, ie. a `eval_results.csv` has data partaining to one hyperparameter tuning trial. In our example above we choose to run 2 tuning trials, but the default for an MNIST workload is 20 tuning trials. Combining all the `eval_results.csv` files across hyperparameter tuning trials is left to users, but a convienence function called `concatenate_csvs()` is provided and demonstrated below. The data format of `eval_results.csv` is designed to be safe to arbitrarily join CSVs without attribute name conflicts across hyperparameter tuning trials or even workloads. It is not done automatically for you because we do not want to create data duplication if there is no need.

You can join all files named `eval_results.csv` in a given folder recursively with this bash command:

```bash
$ python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOGGING_DIR', output_name='all_eval_results.csv')"
```

### 8. (Optional) Add arbitrary key/values to your measurement data
You can also specify arbitrary extra metadata to be saved alongside the output results. This is useful when doing multiple needing a way to tell data apart. To do this use the option `--extra_metadata="key=value"`. You can specify this option multiple times. Choose a unique key that is not likely to overlap with other CSV/JSON data attributes.

### 9. (Optional) Change the frequency of model evaluation

You can override the default frequency of model evaluation, which in turn will change when information about the training progress is saved to disk. This is not competition legal but can be used to monitor training progress at any granularity for debugging purposes. By default the competition evaluates the model every "eval_period_time_sec" seconds, but instead you can choose an eval frequency in epochs or steps. These evals contribute to the accumulated_submission_time.

Example usage:
- Evaluate after every epoch: --eval_frequency_override="1 epoch"
- Evaluate after 100 mini-batches: --eval_frequency_override="100 step"

Note: This option requires `--logging_dir` to set to take effect.

### 10. (Optional) Write a simple plotting script to visualize the results.

For example this minimal plot script:

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read Data
input_file = 'eval_results.csv'
df = pd.read_csv(input_file)

# Plot
sns.set_theme()
fig, ax = plt.subplots()
sns.lineplot(data=df, ax=ax, x='global_step', y='loss')

# Style
ax.set_ylabel('Loss')
ax.set_xlabel('Global Step')

# Save
fig.savefig('plot_loss.png', transparent=False, dpi=160, bbox_inches="tight")
```

Will produce the following loss curve:

<img src="https://gist.githubusercontent.com/danielsnider/73519752010e430af55380f9bf2ee653/raw/fa22c7f012cb1423f5579fc4a0470fa7cecf49a0/plot_loss.png" height="350px">
