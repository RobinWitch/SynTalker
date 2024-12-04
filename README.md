
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enabling-synergistic-full-body-control-in-1/gesture-generation-on-beat2)](https://paperswithcode.com/sota/gesture-generation-on-beat2?p=enabling-synergistic-full-body-control-in-1)
  <a href='https://huggingface.co/spaces/robinwitch/SynTalker'>
  <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow'></a>
<a href="https://colab.research.google.com/drive/1fEXgO-6wGYxfnFuqT12wUdk8-XpHx6yQ?usp=sharing" 
target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Syntalker Colab"/></a> 
# SynTalker: Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation

<center>
  <a href="https://robinwitch.github.io/SynTalker-Page">Project Page</a> •
  <a href="https://arxiv.org/abs/2410.00464">Arxiv Paper</a> •
  <a href="https://youtu.be/hkCQLrLarxs">Demo Video</a> •
  <a href="https://huggingface.co/spaces/robinwitch/SynTalker">Web Gradio Demo</a> •
  <a href="https://colab.research.google.com/drive/1fEXgO-6wGYxfnFuqT12wUdk8-XpHx6yQ?usp=sharing">Colab</a> •
  <a href="#-citation">Citation</a>
</center>

# 📝 Release Plans

- [x] A simple and powerful cospeech model (corespond to paper Table2:SynTalker (w/o both) )
- [x] Training scripts (include training rvqvae and diffusion)
- [x] A web demo (We strongly suggest you to try it!)
- [x] Our syntalker can recieve both speech and text input simultaneously
- [ ] Training scripts (include data preprocessing, training rvqvae, text-motion alignspace and diffusion)

# 💖 Online Demo
Thank Hugging Face🤗 for providing us GPU! Feel free to exprience our online [web demo](https://huggingface.co/spaces/robinwitch/SynTalker)!


# ⚒️ Installation

## Build Environtment

```
conda create -n syntalker python=3.12
conda activate syntalker
pip install -r requirements.txt
bash demo/install_mfa.sh
```


## Download Model
```
gdown https://drive.google.com/drive/folders/1tGTB40jF7v0RBXYU-VGRDsDOZp__Gd0_?usp=drive_link -O ./ckpt --folder
gdown https://drive.google.com/drive/folders/1MCks7CMNBtAzU2XihYezNmiGT_6pWex8?usp=drive_link -O ./datasets/hub --folder
```

## Download Dataset
> For evaluation and training, not necessary for running a web demo or inference.

- Download the original raw data
```
bash preprocess/bash_raw_cospeech_download.sh
```

# 🚩 Running
## Run a web demo
```
python demo.py -c ./configs/diffusion_rvqvae_128_hf.yaml
```

**Notice**: 
If you use ssh to conect and run code in a headless computer, you may encounter an error `pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"`. 
Here, we recommend a method to solve it.

```
sudo apt-get install libegl1-mesa-dev libgles2-mesa-dev
PYOPENGL_PLATFORM='egl' python demo.py -c ./configs/diffusion_rvqvae_128_hf.yaml
```


## Eval
> Require download dataset 
```
python test.py -c configs/diffusion_rvqvae_128.yaml
```

We also provide a [colab notebook](https://colab.research.google.com/drive/1fEXgO-6wGYxfnFuqT12wUdk8-XpHx6yQ?usp=drive_link) for you to evaluate it.

# 📺 Visualize
Following [EMAGE](https://github.com/PantoMatrix/PantoMatrix), you can download [SMPLX blender addon](https://drive.google.com/file/d/1O04GfzUw73PkPBhiZNL98vXpgFjewFUy/view?usp=drive_link), and install it in your blender 3.x or 4.x. Click the button `Add Animation` to visualize the generated smplx file (like xxx.npz).


# 🔥 Training from scratch

## 1. Train RVQVAE

> Well, if your multiple gpus, we can parellel run these three commands.

```
python rvq_beatx_train.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --code-dim 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part upper
```


```
python rvq_beatx_train.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --code-dim 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part hands
```

```
python rvq_beatx_train.py --batch-size 256 --lr 2e-4 --total-iter 300000 --lr-scheduler 200000 --nb-code 512 --code-dim 512 --down-t 2 --depth 3 --dilation-growth-rate 3 --out-dir outputs/rvqvae --vq-act relu --quantizer ema_reset --loss-vel 0.5 --recons-loss l1_smooth --exp-name RVQVAE --body_part lower_trans
```

## 2. Train Diffusion Model

```
python train.py -c configs/diffusion_rvqvae_128.yaml
```


# 💭 Enabling Text Control

## Download models and datasets
```
python preprocess/download_hf.py
```

## Generate
> It will generate motions using the prompt you input and the speech audio from test datasets
```
python test_h3d.py -c configs/diffusion_h3d.yaml --upper_prompt "your prompt" --lower_prompt "your prompt"
```

For example:
```
python test_h3d.py -c configs/diffusion_h3d.yaml --upper_prompt "A person raises up right hand" --lower_prompt "A man is kneel down"
```
The path of the generated motion npy file, the rendered video file, and the audio used,  will be automatically printed in the console.

If you want to further convert it into a npz file format that can be imported into Blender for visualization, you can proceed as follows:

First you should install submodules,
```
git submodule update --init --recursive
git submodule update --remote --recursive
```
Then run the command,
```
python libs/human_body_prior/tutorials/mdm_motion2smpl.py --input your_npy_file_path --output your_file_name.npz
```
For example:
```
python libs/human_body_prior/tutorials/mdm_motion2smpl.py --input ./outputs/audio2pose/test/custom/1128_151129_diffusion_h3d/999/rec_0_prompt.npy --output your_file_name.npz
```
## Training (TO DO)

### 1. Datapreprocess
We need to download AMASS datasets(both smlph and smplx format) from https://amass.is.tue.mpg.de/
into `./datasets`.

We also need to download BEATX datasets as same as previous.
```
bash preprocess/bash_raw_cospeech_download.sh
```
### 2. FPS Correction
The FPS for all motions in AMASS SMPLX is currently set to 120, which is incorrect and needs to be adjusted. Meanwhile, the FPS for SMPLH is correct, so we need to align all SMPLX FPS values with the corresponding SMPLH FPS values.

Following [SMPLX_FPS_Correction.ipynb](preprocess/SMPLX_FPS_Correction.ipynb) to correct it.

### 3. Get HumanML3D Representation
TODO

### 4. Train RVQVAE
TODO

### 5. Train Text-Motion-Align-Space
TODO

### 6. Train Diffusion Model
We also provide a data cache for training it. Data cache is only made up of data's HumanML3D representation, and not rely on RVQVAE and Text-Motion-Align-Space.

Make sure you have [Download models and datasets](#download-models-and-datasets), and unziped them.(include `datasets/HumanML3D`, `datasets/beat_cache/beatx_1-30_amass_h3d.zip`)

Then just run the command as below.
```
python train_h3d.py -c configs/diffusion_h3d.yaml --use_amass True
```



# 🙏 Acknowledgments
Thanks to [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024), [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MoMask](https://github.com/EricGuo5513/momask-codes), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [TMR](https://github.com/Mathux/TMR), [OpenTMA](https://github.com/LinghaoChan/OpenTMA), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [human_body_prior](https://github.com/nghorbani/human_body_prior), our code is partially borrowing from them. Please check these useful repos.


# 📖 Citation

If you find our code or paper helps, please consider citing:

```bibtex
@inproceedings{chen2024syntalker,
  author = {Bohong Chen and Yumeng Li and Yao-Xiang Ding and Tianjia Shao and Kun Zhou},
  title = {Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation},
  booktitle = {Proceedings of the 32nd ACM International Conference on Multimedia},
  year = {2024},
  publisher = {ACM},
  address = {New York, NY, USA},
  pages = {10},
  doi = {10.1145/3664647.3680847}
}
```