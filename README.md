# KUIELab-MDX-Net

## 0. Environment

- Ubuntu 20.04
- at least four cuda-able GPUs (each > 2080ti)
- 1.5 TB disk storage for data augmentation
- wandb for logging

Also, you ***must*** create .env file by copying .env.sample to set environmental variables.

```
wandb_api_key=[Your Key] # "xxxxxxxxxxxxxxxxxxxxxxxx"
data_dir=[Your Path] # "/home/ielab/repos/musdbHQ"
```

- about ```wandb_api_key```
   - we currently only support wandb for logging.
   - for ```wandb_api_key```, visit [wandb](https://wandb.ai/site), go to ```setting```, and then copy your api key
- about ```data_dir```
   - the ***absolute*** path where datasets are stored

## 1. Installation

```bash
conda env create -f conda_env_gpu.yaml -n mdx-net
conda activate mdx-net
pip install -r requirements.txt
sudo apt-get install soundstretch
```

## 2. Training & Submission

- [Leaderboard_A](https://github.com/kuielab/mdx-net/tree/Leaderboard_A)
- [Leaderboard_B](https://github.com/kuielab/mdx-net/tree/Leaderboard_B)


# ACKNOWLEDGEMENT

- This repository is based on [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template)
- Repository of [TFC-TDF-U-Net](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS), our previous ISMIR 2020 paper 
- Also, facebook/[demucs](https://github.com/facebookresearch/demucs)
