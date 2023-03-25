# A Self-supervised-driven Open-set Unsupervised Domain Adaptation Method for Optical Remote Sensing Image Scene Classification and Retrieval
This repository is the official implementation of [A Self-supervised-driven Open-set Unsupervised Domain Adaptation Method for Optical Remote Sensing Image Scene Classification and Retrieval](https://ieeexplore.ieee.org/document/10078892) (IEEE TGRS 2023).


<b>Authors</b>: Siyuan Wang, Dongyang Hou and Huaqiao Xing


## Requirements
- This code is written for `python3`.
- pytorch >= 1.7.0
- torchvision
- numpy, prettytable, tqdm, scikit-learn, matplotlib, argparse, h5py


## Data Preparing
Download dataset from the following link (code is chk8):

[BaiduYun](https://pan.baidu.com/s/1YbsJZQEFaLyl3HRE3uBsbQ)

## Training and Evaluating
The pipeline for training with SSOUDA is the following:

1. Train the model. For example, to run an experiment for UCM_LandUse dataset (source domain) and AID dataset (target domain),  run:

`python ssouda.py /your_path/SSOUDA_dataset/ -s UCMD -t NWPU -a resnet50 --epochs 60 --seed 1 --log logs/ucmd_nwpu`

2. Evaluate the PCLUDA network:

`python ssouda.py /your_path/SSOUDA_dataset/ -s UCMD -t NWPU -a resnet50 --epochs 60 --seed 1 --log logs/ucmd_nwpu --phase test`

## Acknowledgment
This code is heavily borrowed from [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)

## Citation
If you find our work useful in your research, please consider citing our paper:

```
@ARTICLE{10078892,
  author={Wang, Siyuan and Hou, Dongyang and Xing, Huaqiao},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Self-supervised-driven Open-set Unsupervised Domain Adaptation Method for Optical Remote Sensing Image Scene Classification and Retrieval}, 
  year={2023},
  doi={10.1109/TGRS.2023.3260873}
}
```
## Contact
Please contact wsy.mail@foxmail.com if you have any question on the codes.
