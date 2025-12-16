# AI707 PROJECT - REIMPLEMENTATION OF RISE

This code based on [Diffusion Policy Policy Optimization (DPPO)](https://github.com/irom-princeton/dppo) 

## Installation 

1. Clone the repository
```console
git clone https://github.com/longdvt/AI707_RISE.git
cd AI707_RISE
```

2. Install dependencies with a conda environment on a Linux machine with a Nvidia GPU.
```console
conda create -n AI707 python=3.8 -y
conda activate AI707
pip install -e .
```

3. Install specific environment dependencies (Robomimic).
```console
pip install -e .[robomimic]
```

4. [Install MuJoCo for Robomimic](installation/install_mujoco.md).

5. Set environment variables for data and logging directory (default is `data/` and `log/`), and set WandB entity (username or team name)
```
source script/set_path.sh
```

## Usage - Pre-training

Pre-training data for all tasks are pre-processed and can be found at [here](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link). Pre-training script will download the data (including normalization statistics) automatically to the data directory.
<!-- The data path follows `${DPPO_DATA_DIR}/<benchmark>/<task>/train.npz`, e.g., `${DPPO_DATA_DIR}/gym/hopper-medium-v2/train.npz`. -->

### Run pre-training with data
All the configs can be found under `cfg/<env>/pretrain/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.

```console
# Robomimic - lift/can/square/transport
# Lift
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/lift

# Can
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/can

# Square
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/square

# Transport
python script/run.py --config-name=pre_diffusion_mlp \
    --config-dir=cfg/robomimic/pretrain/transport
```

## Usage - Evaluation
Pre-trained or fine-tuned policies can be evaluated without running the fine-tuning script now. Some example configs are provided under `cfg/{gym/robomimic/furniture}/eval}` including ones below. Set `base_policy_path` to override the default checkpoint, and `ft_denoising_steps` needs to match fine-tuning config (otherwise assumes `ft_denoising_steps=0`, which means evaluating the pre-trained policy). We additionally add script to log non-expert rollouts, used for training RISE later.
```console
# Robomimic - lift/can/square/transport
# Lift
python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/robomimic/eval/lift/

# Can
python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/robomimic/eval/can/

# Square
python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/robomimic/eval/square/

# Transport
python script/run.py --config-name=eval_diffusion_mlp --config-dir=cfg/robomimic/eval/transport/
```

See [here](script/process_dataset.py) to see how we process the dataset to train RISE.

## Usage - Training RISE
```console
# Robomimic - lift/can/square/transport
# Lift
python script/run.py --config-name=offline_idql_diffusion_mlp --config-dir=cfg/robomimic/offline/lift/

# Can
python script/run.py --config-name=offline_idql_diffusion_mlp --config-dir=cfg/robomimic/offline/can/

# Square
python script/run.py --config-name=offline_idql_diffusion_mlp --config-dir=cfg/robomimic/offline/square/

# Transport
python script/run.py --config-name=offline_idql_diffusion_mlp --config-dir=cfg/robomimic/offline/transport/
```

## License
This repository is released under the MIT license. See [LICENSE](LICENSE).

## Acknowledgement
* [DPPO, Allen et al.](https://github.com/irom-princeton/dppo): General code base
* [Robomimic, Mandlekar et al.](https://github.com/ARISE-Initiative/robomimic): Robomimic benchmark
* [IDQL, Hansen-Estruch et al.](https://github.com/philippe-eecs/IDQL): IDQL baseline