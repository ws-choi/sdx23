![Banner image](https://images.aicrowd.com/uploads/ckeditor/pictures/1040/content_Desktop_Banner.png)

# **[Music Demixing Challenge 2023 - Music Separation](https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23)** - Starter Kit - KUIELab-Mdx-Net Edition
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/fNRrSvZkry)

This repository is the Sound Demixing Challenge 2023 - Music Separation **Starter kit**, KUIELab-MDX-Net Edition ! It contains:
*  **How to train** a vanilla KUIELab-MDX-Model for the challenge!
*  **How to submit** a trained KUIELab-MDX-Model to AICrowd's system

This repository does not cover 

* How to make training of KUIELab-MDX-Net more robust on corrupted dataset
* Training [Mixer](https://arxiv.org/abs/2111.12203), which was used in the previous challenge was omitted.


## about KUIELab-MDX-Net

- [presentation slide](https://ws-choi.github.io/personal/presentations/slide/2021-08-21-aicrowd)

## A. Quick Start

If you want to submit the pretrained baseline, please click [here](https://gitlab.aicrowd.com/Woosung.Choi.Sony/sdx-2023-music-demixing-track-starter-kit)!

## B. How to reproduce this model?
### 1. Environment

- Ubuntu 20.04
- at least two cuda-able GPUs (each >= 2080ti)
- wandb or tensorboard for logging

Also, you ***must*** create .env file by copying .env.sample to set environmental variables.

```
# Download datasets and link paths to these variables
# https://www.aicrowd.com/challenges/sound-demixing-challenge-2023/problems/music-demixing-track-mdx-23/dataset_files

label_noise_data_dir="/home/git/moisesdb23_labelnoise_v1.0/"
bleeding_data_dir="/home/git/moisesdb23_bleeding_v1.0/"

# You don't have to fill this if you want to use tensorboard only.
wandb_api_key=[YOUR WANDB API KEY] # go wandb.ai/settings and copy your key
```

- about ```wandb_api_key```
   - we currently support wandb and tensorboard for logging.
   - for ```wandb_api_key```, visit [wandb](https://wandb.ai/site), go to ```setting```, and then copy your api key
- about ```label_noise_data_dir``` and ```bleeding_data_dir```
   - the ***absolute*** path where `moisesdb23_*_v*` datasets are stored

### 2. Installation

```bash
conda env create -f conda_env_gpu.yaml -n mdx-net-sdx23
conda activate mdx-net-sdx233
pip install -r requirements.txt
```

### 3. Training

- Train four different models: `vocals`, `drums`, `bass` and `other`

#### A. for label_noise
   - vocals:
     - `python train.py experiment=multigpu_vocals datamodule=moisesdb23_labelnoise_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`
   - drums:
     - `python train.py experiment=multigpu_drums datamodule=moisesdb23_labelnoise_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`
   - bass:
     - `python train.py experiment=multigpu_bass datamodule=moisesdb23_labelnoise_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`
   - other:
     - `python train.py experiment=multigpu_other datamodule=moisesdb23_labelnoise_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`

#### B. for bleeding
   - vocals:
     - `python train.py experiment=multigpu_vocals datamodule=moisesdb23_bleeding_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`
   - drums:
     - `python train.py experiment=multigpu_drums datamodule=moisesdb23_bleeding_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`
   - bass:
     - `python train.py experiment=multigpu_bass datamodule=moisesdb23_bleeding_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`
   - other:
     - `python train.py experiment=multigpu_other datamodule=moisesdb23_bleeding_v1 trainer.gpus=2 logger=tensorboard trainer=ddp`

#### Other options

##### gpus
- you can use single gpu by setting `trainer.gpus=1`
- you can specify the ids of gpus that you want to use `trainer.gpus=[1,3]`

##### loggers
- you can use tensorboard by setting `logger=tensorboard`
- you can use wandb by setting `logger=wandb`
- you can use both of them by setting `logger=['tensorboard,wandb']`

##### batch_sizes
- you can adjust the size of batch per each node: `datamodule.batch_size=2`

##### Model Details
You can change hyperparameters if you want in `configs/model/ConvTDFNet_bass.yaml`, `configs/model/ConvTDFNet_drums.yaml`, `configs/model/ConvTDFNet_other.yaml`, `configs/model/ConvTDFNet_vocals.yaml`

### 4. Submission

- If you think there is no sdr improvement in validation score, then please stop training.
- During training, `configs/callbacks/default.yaml` defines `model_checkpoint` automatically, which stores the checkpoints with the top-5 sdrs.
  - you can set different value for it in `configs/callbacks/default.yaml` or putting `callbacks.model_checkpoint.save_top_k=10` to the shell command.
- check your checkpoints. for example, 
    
    ```commandline
    ls logs/runs/vocals/2023-02-27/16-52-17/checkpoints/
    'epoch=00.ckpt'   last.ckpt    
    ```
#### A. for label_noise

1. Please fork this [repository](https://gitlab.aicrowd.com/Woosung.Choi.Sony/sdx-2023-music-demixing-track-starter-kit) and set up.
    ```commandline
    git clone https://gitlab.aicrowd.com/[YourGitLabAccount]/sdx-2023-music-demixing-track-starter-kit
    cd sdx-2023-music-demixing-track-starter-kit
    git checkout label_noise
    ls models/mdx-net-labelnoise/vocals/
    ```
2. it should give
    ```commandline
    config.yaml  epoch=2543.ckpt
    ```
3. you can add your own checkpoints, like this

```commandline
config.yaml  epoch=2543.ckpt epoch=00.ckpt
```

4. open `sdx-2023-music-demixing-track-starter-kit/my_submission/mdxnet_labelnoise_music_separation_model.py` and link checkpoints.

```python
    def __init__(self):
        self.separator = torch.nn.ModuleDict(
            {
                'vocals': get_model(root_dir='./models/mdx-net-labelnoise/vocals', ckpt_path='epoch=00.ckpt'),
                # 'vocals': get_model(root_dir='./models/mdx-net-labelnoise/vocals', ckpt_path='epoch=2543.ckpt'),
                'bass': get_model(root_dir='./models/mdx-net-labelnoise/bass', ckpt_path='epoch=694.ckpt'),
                'drums': get_model(root_dir='./models/mdx-net-labelnoise/drums', ckpt_path='epoch=408.ckpt'),
                'other': get_model(root_dir='./models/mdx-net-labelnoise/other', ckpt_path='epoch=763.ckpt')
            }
        )
```

5. don't forget use git-lfs to upload them to your own repository

```commandline
git lfs track models/mdx-net-labelnoise/vocals/epoch\=00.ckpt
git add .gitattributes
```

6. commit and submit it to aicrowd!

```commandline
git tag -am "submission-mdx-label-noise" "submission-mdx-label-noise" 
git push
```

#### B. for bleeding

1. Please fork this [repository](https://gitlab.aicrowd.com/Woosung.Choi.Sony/sdx-2023-music-demixing-track-starter-kit) and set up.
    ```commandline
    git clone https://gitlab.aicrowd.com/[YourGitLabAccount]/sdx-2023-music-demixing-track-starter-kit
    cd sdx-2023-music-demixing-track-starter-kit
    git checkout bleeding
    ls models/mdx-net-bleeding/vocals/
    ```
2. it should give
    ```commandline
    config.yaml  epoch=744.ckpt
    ```
3. you can add your own checkpoints, like this

```commandline
config.yaml  epoch=744.ckpt epoch=00.ckpt
```

4. open `sdx-2023-music-demixing-track-starter-kit/my_submission/mdxnet_bleeding_music_separation_model.py` and link checkpoints.

```python
    def __init__(self):
        self.separator = torch.nn.ModuleDict(
            {
                'vocals': get_model(root_dir='./models/mdx-net-labelnoise/vocals', ckpt_path='epoch=00.ckpt'),
                #'vocals': get_model(root_dir='./models/mdx-net-bleeding/vocals', ckpt_path='epoch=744.ckpt'),
                'bass': get_model(root_dir='./models/mdx-net-bleeding/bass', ckpt_path='epoch=995.ckpt'),
                'drums': get_model(root_dir='./models/mdx-net-bleeding/drums', ckpt_path='epoch=780.ckpt'),
                'other': get_model(root_dir='./models/mdx-net-bleeding/other', ckpt_path='epoch=437.ckpt')
            }
        )
```

5. don't forget use git-lfs to upload them to your own repository

```commandline
git lfs track models/mdx-net-bleeding/vocals/epoch\=00.ckpt
git add .gitattributes
```

6. commit and submit it to aicrowd!

```commandline
git tag -am "submission-mdx-bleeding" "submission-mdx-bleeding" 
git push
```

# ACKNOWLEDGEMENT

- This repository is based on [Lightning-Hydra Template](https://github.com/ashleve/lightning-hydra-template)
- Repository of [TFC-TDF-U-Net](https://github.com/ws-choi/ISMIR2020_U_Nets_SVS), our previous ISMIR 2020 paper 
