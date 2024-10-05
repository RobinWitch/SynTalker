
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enabling-synergistic-full-body-control-in-1/gesture-generation-on-beat2)](https://paperswithcode.com/sota/gesture-generation-on-beat2?p=enabling-synergistic-full-body-control-in-1)

# SynTalker: Enabling Synergistic Full-Body Control in Prompt-Based Co-Speech Motion Generation
>
<center>
  <a href="https://robinwitch.github.io/SynTalker-Page">Project Page</a> •
  <a href="https://arxiv.org/abs/2410.00464">Arxiv Paper</a> •
  <a href="(https://www.youtube.com/watch?v=hkCQLrLarxs&t=7s)">Demo Video</a> •
  <a href="#-citation">Citation</a>
</center>

# 📝 Release Plans

- [x] A simple and powerful cospeech model (corespond to paper Table2:SynTalker (w/o both) )
- [x] Training scripts (include training rvqvae and diffusion)
- [x] A web demo (We strongly suggest you to try it!)
- [ ] Our syntalker can recieve both speech and text input simultaneously
- [ ] Training scripts (include data preprocessing, training rvqvae, text-motion alignspace and diffusion)

# ⚒️ Installation

## Build Environtment

```
conda create -n syntalker python=3.12
conda activate syntalker
pip install -r requirements.txt
```


## Download Data

We provide two ways for getting data, if you want to quickly run this project, you can choose `Download the parsed data directly`, or if you want to build datasets from raw data, you can choose `Download the original raw data`.
- Download the parsed data directly
```
bash bash_cospeech_download.sh
```

- Download the original raw data
```
bash bash_raw_cospeech_download.sh
```
## Eval
- Evaluate metric
```
python test.py -c configs/diffusion_rvqvae_128.yaml
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
export PYOPENGL_PLATFORM='egl'
python demo.py -c ./configs/diffusion_rvqvae_128_hf.yaml
```

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


# Acknowledgments
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