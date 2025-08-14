# Setting Up Programming Environment


In order to run hands-on exercises in this course, you need the Python package and several depenencies.
- If you use your own computer to run exercises, you should follow the instructions described below to install relevant packages and setup specific programming environment before running hands-on exercises.
- You can use an HPC cluster if you have access to one to run hands-on exercises. Below we provide a short description to login to the [**LUMI**](https://www.lumi.csc.fi/public/) cluster, load the modules, and run interactive/batch jobs.

## Local Installation

### Install miniforge

If you already have a preferred way to manage Python versions and libraries, you can stick to that. If not, we recommend that you install Python3 and all libraries using Miniforge, a free minimal installer for the package, dependency and environment manager conda.

Please follow the [installation instructions](https://conda-forge.org/download/) to install Miniforge.

Make sure that conda is correctly installed:
```shell
$ conda --version
# conda 24.11.2
````

### Install python programming environment on personal computer

**For Mac users**

With conda installed, install the required dependencies by running:
```shell
$ conda env create --yes -f https://raw.githubusercontent.com/ENCCS/hpda-python/main/content/env/environment.yml
```

This will create a new environment pyhpda which you need to activate by:
```shell
$ conda activate pyhpda
```

Ensure that the Python version is fairly recent:
```shell
$ python --version
# Python 3.12.8
```

Finally, open Jupyter-Lab in your browser:
```shell
$ jupyter-lab
```

If you use VS code, you can come to the installed `pyhpda` programming environment via choosing `Select Kernel` ar the upper right corner, `Python Environents` and you will find the pre-installed `pyhpda` programming environment.

**For Linux users**

Please provide detailes instructures for linux users to install packages.

**For Windows users**

Please provide detailes instructures for windows users to install packages.

## Using HPC Cluster

### LUMI

#### Login to LUMI cluster

Follow practical instructions [**HERE**](https://enccs.se/tutorials/2024/02/log-in-to-lumi-cluster/) to get your access to LUMI cluster.
- On Step 5, you can login to LUMI cluster through terminal.
- On Step 6, you can login to LUMI cluster from the web-interface.

#### Running jobs on LUMI cluster

If you want to run an interactive job asking for 1 node, 1 GPU, and 1 hour:
```bash
$ salloc -A project_XXXXX -N 1 -t 1:00:00 -p standard-g --gpus-per-node=1

$ srun <some-command>
```

Exit interactive allocation with `exit`.

You can also submit your job with a batch script submit.sh:

```bash
#!/bin/bash -l
#SBATCH --account=project_XXXXX
#SBATCH --job-name=example-job
#SBATCH --output=examplejob.o%j
#SBATCH --error=examplejob.e%j
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1:00:00

srun <some_command>
```

Some useful commands are listed below:
- Submit the job: `sbatch submit.sh`
- Monitor your job: `squeue --me`
- Kill job: `scancel <JOB_ID>`

#### Using `pyhpda` programming environment on LUMI cluster

We have installed the `pyhpda` programming environment on LUMI. You can follow instructions below to activate it and login to LUMI cluster, either via terminal or through the web-interface.

**Login to LUMI cluster via terminal** and then the commands below to check and activate the pyhpda environment.
```bash
$ /projappl/projecXXXXX10/miniconda3/bin/conda init

$ source ~/.bashrc

$ which
# you should get output as shown below
/project/project_XXXXX/miniconda3/condabin/cond

$ conda activate pyhpda

$ which python
# you should get output as shown below
/project/project_XXXXX/miniconda3/envs/pyhpda/bin/python
 conda
```

**Login to LUMI cluster via [web-interface]()** and then select `Jupyter` (not `Jupyter for courses`) icon for an interactive session, and provide the following values in the form to launch the jupyter lab app.
- Project: `project_XXXXX`
- Partition: `interactive`
- Number of CPU cores: `2`
- Time: `4:00:00`
- Working directory: `/projappl/project_XXXXX`
- Python: `Custom`
- Path to python: `/project/project_XXXXX/miniconda3/envs/pyhpda/bin/python`
- `check` for *Enable system installed packages on venv creation*
- `check` for Enable packages under *~/.local/lib on venv start*
- Click the `Launch` button, wait for minutes until your requested session was created.
- Click the `Connect to Jupyter` button, and then select the Python kernel `Python 3 (venv)` for the created Jupyter notebooks.

### Leonardo Booster

#### Login to Leonardo Booster cluster

Follow instructions at [HERE](https://enccs.se/news/2023/09/how-to-login-to-leonardo-supercomputer/) to get your access to Leonardo Booster cluster.

#### Running jobs on Leonardo Booster cluster

Here are instructions to run jobs on Leonardo Booster cluster
- xxx
- xxx

#### Using `pyhpda` programming environment on Leonardo Booster cluster

Here are instructions to install the `pyhpda` programming environment on Leonardo Booster cluster
- xxx
- xxx
