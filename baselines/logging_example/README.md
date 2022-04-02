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
| **Tuning search space** (if used)                 | ✔️                    |                     |                  |
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
- Run a workload with the `--logging_dir` option to produce a `measurements.csv` and `metadata.json` files.
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
./logs/[workload_name]/metadata.json
./logs/[workload_name]/packages.txt
./logs/[workload_name]/trial_[n]/metadata.json
./logs/[workload_name]/trial_[n]/measurements.csv
```

What are these output files?

Four files are written to the `logging_dir` folder:
  1. `metadata.json` is created at the start of a workload and it includes the datetime, workload name, and system configuration.
  1. `packages.txt` is created at the start of a workload and it includes a list of the currently installed OS and python packages.
  1. `trial_[n]/measurements.csv` is created for each hyperparameter tuning trial and a row is appended for every model evaluation. The information included is loss, accuracy, training step, time elapsed, hparams, workload properties, and hardware utilization.
  1. `trial_[n]/metadata.json` is created at the end of each hyperparameter tuning trial and includes the result of the trial. The last row of measurements.csv and this file are very similar but can differ in the number of steps and runtime in one situation: when the tuning trial ran out of time but completed one or more steps after the last model evaluation.

### 3. Inspect the output of `metadata.json`

```
{
    "workload": "mnist_jax",
    "datetime": "2022-03-12T00:31:13.366143",
    "status": "COMPLETE",
    "score": 3.9674811363220215,
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
    "workload.num_eval_examples": 10000,
    "workload.num_train_examples": 60000,
    "workload.target_value": 0.9,
    "workload.train_mean": 0.1307,
    "workload.train_stddev": 0.3081,
    "os_platform": "Linux-5.4.48-x86_64-with-glibc2.29",
    "python_version": "3.8.10",
    "python_compiler": "GCC 9.3.0",
    "git_branch": "main",
    "git_commit_hash": "d062ee3b96a5bd342783ad5dd461cfa86d635124",
    "cpu_model_name": "AMD EPYC 7601 32-Core Processor",
    "cpu_count": 64,
    "gpu_model_name": "GeForce RTX 2080 Ti",
    "gpu_count": 4,
    "gpu_driver": "460.27.04"
}
```

### 4. Inspect the output of `measurements.csv` (only one row shown)

|workload |trial_idx|datetime                  |accumulated_submission_time|accuracy          |loss             |global_step|epoch|steps_per_epoch|global_start_time |goal_reached|is_time_remaining|training_complete|batch_size|hparam.learning_rate|hparam.one_minus_beta_1|hparam.epsilon|workload.eval_period_time_sec|workload.max_allowed_runtime_sec|workload.num_eval_examples|workload.num_train_examples|workload.target_value|workload.train_mean|workload.train_stddev|cpu.util.avg_percent_since_last|cpu.freq.current |temp.k10temp.current|mem.total   |mem.available|mem.used  |mem.percent_used|mem.read_bytes_since_boot|mem.write_bytes_since_boot|net.bytes_sent_since_boot|net.bytes_recv_since_boot|gpu.count|gpu.0.compute.util|gpu.0.mem.util    |gpu.0.mem.total|gpu.0.mem.used|gpu.0.mem.free|gpu.0.temp.current|gpu.1.compute.util|gpu.1.mem.util   |gpu.1.mem.total|gpu.1.mem.used|gpu.1.mem.free|gpu.1.temp.current|gpu.2.compute.util|gpu.2.mem.util   |gpu.2.mem.total|gpu.2.mem.used|gpu.2.mem.free|gpu.2.temp.current|gpu.3.compute.util|gpu.3.mem.util   |gpu.3.mem.total|gpu.3.mem.used|gpu.3.mem.free|gpu.3.temp.current|gpu.avg.compute.util|gpu.avg.mem.util  |gpu.avg.mem.total|gpu.avg.mem.used|gpu.avg.mem.free|gpu.avg.temp.current|
|---------|---------|--------------------------|---------------------------|------------------|-----------------|-----------|-----|---------------|------------------|------------|-----------------|-----------------|----------|--------------------|-----------------------|--------------|-----------------------------|--------------------------------|--------------------------|---------------------------|---------------------|-------------------|---------------------|-------------------------------|-----------------|--------------------|------------|-------------|----------|----------------|-------------------------|--------------------------|-------------------------|-------------------------|---------|------------------|------------------|---------------|--------------|--------------|------------------|------------------|-----------------|---------------|--------------|--------------|------------------|------------------|-----------------|---------------|--------------|--------------|------------------|------------------|-----------------|---------------|--------------|--------------|------------------|--------------------|------------------|-----------------|----------------|----------------|--------------------|
|mnist_jax|1        |2022-03-12T00:30:58.435387|2.829345464706421          |0.0900000035762786|2.465173721313477|0          |0.0  |58             |1647045052.9468262|False       |True             |False            |1024      |0.0001880581412603  |0.9261041856291228     |1e-08         |10                           |60                              |10000                     |60000                      |0.3                  |0.1307             |0.3081               |3.3                            |2.683687500000002|32.625              |135058636800|124791758848 |8878366720|7.6             |77709185505280           |3078489495552             |2949781                  |76184472                 |4        |0.0               |0.8207641346764679|11019.0        |9044.0        |1975.0        |34.0              |0.0               |0.818223069244033|11019.0        |9016.0        |2003.0        |35.0              |0.0               |0.818223069244033|11019.0        |9016.0        |2003.0        |36.0              |0.0               |0.818223069244033|11019.0        |9016.0        |2003.0        |38.0              |0.0                 |0.8188583356021417|11019.0          |9023.0          |1996.0          |35.75               |


### 5. (Optional) Combine trial JSON files if you want the measurements at the end of each trial in 1 file

You can join all files named `trial_[n]/metadata.json` in a given folder recursively with this bash command:

```bash
$ jq -s 'flatten' ./logs*/**/trial_**/*.json > all_metadata.json
```

### 6. (Optional) Combine trial CSV files if you want all measurements throughout all trials in 1 file

By default, one `measurements.csv` is produced per training run, ie. a `measurements.csv` has data partaining to one hyperparameter tuning trial. In our example above we choose to run 2 tuning trials, but the default for an MNIST workload is 20 tuning trials. Combining all the `measurements.csv` files across hyperparameter tuning trials is left to users, but a convienence function called `concatenate_csvs()` is provided and demonstrated below. The data format of `measurements.csv` is designed to be safe to arbitrarily join CSVs without attribute name conflicts across hyperparameter tuning trials or even workloads. It is not done automatically for you because we do not want to create data duplication if there is no need.

You can join all files named `measurements.csv` in a given folder recursively with this bash command:

```bash
$ python3 -c "from algorithmic_efficiency import logging_utils; logging_utils.concatenate_csvs('$LOGGING_DIR')"
````

This will produce a file called:
```bash
./logs/all_measurements.csv
```

### 7. (Optional) Add arbitrary key/values to your measurement data
You can also specify arbitrary extra metadata to be saved alongside the output CSVs measurements and JSON metadata. This is useful when doing multiple needing a way to tell data apart. To do this use the option `--extra_metadata="key=value"`. You can specify this option multiple times. Choose a unique key that is not likely to overlap with other CSV/JSON data attributes.

### 8. (Optional) Change the frequency of model evaluation

You can override the default frequency of model evaluation, which in turn will change when information about the training progress is saved to disk. This is not competition legal but can be used to monitor training progress at any granularity for debugging purposes. By default the competition evaluates the model every "eval_period_time_sec" seconds, but instead you can choose an eval frequency in epochs or steps. These evals contribute to the accumulated_submission_time.

Example usage:
- Evaluate after every epoch: --eval_frequency_override="1 epoch"
- Evaluate after 100 mini-batches: --eval_frequency_override="100 step"

Note: This option requires `--logging_dir` to set to take effect.

### 9. (Optional) Write a simple plotting script to visualize the results.

For example this minimal plot script:

```python
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Read Data
input_file = './logs/all_measurements.csv'
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
