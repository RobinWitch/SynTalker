
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enabling-synergistic-full-body-control-in/gesture-generation-on-beat2)](https://paperswithcode.com/sota/gesture-generation-on-beat2?p=enabling-synergistic-full-body-control-in)

# 📝 Release Plans

- [x] A simple and powerful cospeech model (corespond to paper Table2:SynTalker (w/o both) )
- [ ] Training scripts (include training rvqvae and diffusion)
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

We provide two ways for getting data, if you want to quickly run this project, you can choose `Download the parsed data directly`, or if you want do build datasets from raw data, you can choose `Download the original raw data`.
- Download the parsed data directly
```
bash bash_cospeech_download.sh
```

- Download the original raw data
```
bash bash_raw_cospeech_download.sh
```
## Eval

```
python test.py -c configs/diffusion_rvqvae_128.yaml
```

# Acknowledgments
Thanks to [EMAGE](https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024), [DiffuseStyleGesture](https://github.com/YoungSeng/DiffuseStyleGesture), [MDM](https://github.com/GuyTevet/motion-diffusion-model), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT), [MoMask](https://github.com/EricGuo5513/momask-codes), [MotionCLIP](https://github.com/GuyTevet/MotionCLIP), [TMR](https://github.com/Mathux/TMR), [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [OpenTMA](https://github.com/LinghaoChan/OpenTMA) , our code is partially borrowing from them. Please check these useful repos.


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