# Environment Setup


- We use Anaconda to install required packages:
    ```bash
        conda env create -f env/env_equiformer.yml
    ```
    This will create a new environment called `equiformer` with Python 3.10, PyTorch 2.3 and CUDA 12.1, validated on both A100 and newer H200 GPUs.

    If your system has a read-only package cache, set `CONDA_PKGS_DIRS` to a directory you own before running the command, e.g.
    `CONDA_PKGS_DIRS=$HOME/conda_pkgs`.

- We activate the environment:
    ```bash
        export PYTHONNOUSERSITE=True    # prevent using packages from base
        conda activate equiformer
    ```

- Besides, [`env/env_equiformer.yml`](../env/env_equiformer.yml) specifies versions of all packages.

- After setting up the environment, clone OC20 repository and install `ocpmodels`:
    ```bash
        git clone https://github.com/FAIR-Chem/fairchem.git
        cd ocp
        git checkout d2aaaeb67687785a99f869b4568e59cbbfab7acb
        pip install -e .
    ```
    The version of `ocpmodels` used here is `0.0.3`. 
    The correpsonding version of GitHub repository is [here](https://github.com/FAIR-Chem/fairchem/tree/d2aaaeb67687785a99f869b4568e59cbbfab7acb).
