This folder contains a SLURM batch script that can be used to run jobs where each job corresponds to a training run on a given workload, training algorithm, random seed and tuning trial (if on external tuning ruleset).

To launch jobs:
1) Generate a job config. The following command will generate a config.json.
```
python3 make_job_config.py \
  --submission_path <submission_path> \
  --tuning_search_space <tuning_search_space> \
  --experiment_dir $HOME/experiments/<algorithm> \
  --framework <jax|pytorch>
```
2) Save the config.json in the same directory you will run the sbatch script from.
3) Check the sbatch script `run_jobs.sh`. 
- Set the task range to the number of tasks in the config.
```
#SBATCH --array=0-119
```
- Set the output and error logs directory for the SLURM logs.
```
#SBATCH --output=experiments/<tuning_ruleset>/<algorithm>/job_%A_%a.out
#SBATCH --error=experiments/<tuning_ruleset>/<algorithm>/job_%A_%a.err
```
4) Submit a SLURM batch job by running:
```
sbatch run_jobs.sh
```
