This folder contains a SLURM batch script that can be used to run jobs where each job corresponds to a training run on a given workload, training algorithm, random seed and tuning trial (if on external tuning ruleset).

To launch jobs:
1) Generate a job config
`python make_job_config.py`