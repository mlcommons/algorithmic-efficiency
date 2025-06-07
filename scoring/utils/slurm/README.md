# Launching SLURM jobs with SBATCH
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


# Set up new SLURM cluster
If you are setting up a new cluster, we recommend using the [HPC toolkit to set up a SLURM cluster](https://cloud.google.com/cluster-toolkit/docs/quickstarts/slurm-cluster).
To set up the new cluster:

1) [Install the Google Cluster Toolkit](https://github.com/GoogleCloudPlatform/cluster-toolkit?tab=readme-ov-file#quickstart). 
2) Create and deploy a packer node to create a base image for the cluster nodes. See [packer builder terraform blueprint](/scoring/utils/slurm/algoperf_slurm_packer_builder.yaml).
3) Manually update the image:
    1) Create a VM from the Disk image created in the previous step.
    2) Install the NVIDIA container toolkit on the VM.
    3) Transfer the data from GCP bucket to `/opt/data`.
    4) Create a new disk image from the VM.
4) Create and deploy the cluster. See [cluster terraform blueprint](/scoring/utils/slurm/algoperf_slurm_cluster.yaml).

