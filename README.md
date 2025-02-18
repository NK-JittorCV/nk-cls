# JittorCLS

## A codebase for image classification based on Jittor


## UPDATE
### Add More models
#### Res2Net: A New Multi-scale Backbone Architecture [TPAMI'21] 

Related links: [[paper]](https://arxiv.org/pdf/1904.01169.pdf) [[中译版全文]](http://mftp.mmcheng.net/Papers/20PAMI-res2net-CN.pdf) [[github]](https://github.com/Res2Net)

```
@article{gao2019res2net,
  title={Res2Net: A New Multi-scale Backbone Architecture},
  author={Gao, Shang-Hua and Cheng, Ming-Ming and Zhao, Kai and Zhang, Xin-Yu and Yang, Ming-Hsuan and Torr, Philip},
  journal={IEEE TPAMI},
  year={2021},
  doi={10.1109/TPAMI.2019.2938758}, 
}
```

#### Pyramid Pooling Transformer for Scene Understanding [TPAMI'22]

Related links: [[paper]](https://mmcheng.net/wp-content/uploads/2022/09/22TPAMI-P2T.pdf) [[中译版全文]](https://mmcheng.net/wp-content/uploads/2022/08/22PAMI_P2T_CN.pdf) [[github]](https://github.com/yuhuan-wu/P2T)

```
@ARTICLE{wu2022p2t,
  author={Wu, Yu-Huan and Liu, Yun and Zhan, Xin and Cheng, Ming-Ming},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={{P2T}: Pyramid Pooling Transformer for Scene Understanding}, 
  year={2022},
  doi = {10.1109/tpami.2022.3202765},
}
```

#### Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition [TPAMI'24]

Related links: [[paper]](https://arxiv.org/abs/2211.11943) [[中译版全文]](https://mftp.mmcheng.net/Papers/24PAMI-Conv2Former-CN.pdf) [[github]](https://github.com/HVision-NKU/Conv2Former)

```
@article{hou2024conv2former,
  title={Conv2Former: A Simple Transformer-Style ConvNet for Visual Recognition},
  author={Hou, Qibin and Lu, Cheng-Ze and Cheng, Ming-Ming and Feng, Jiashi},
  journal={IEEE TPAMI},
  year={2024},
  doi={10.1109/TPAMI.2024.3401450}, 
}
```



## Getting Started
### Install Jittor
```shell
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
# If your computer contains an Nvidia graphics card, check the cudnn acceleration library
python3.7 -m jittor.test.test_cudnn_op
```
For more information on how to install jittor, you can check [here](https://cg.cs.tsinghua.edu.cn/jittor/download/).

### Install OpenMPI
```shell
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```
To obtain more information about OpenMPI, you can check [here](https://www.open-mpi.org/faq/?category=building#easy-build).

### Train
We provide scripts for single-machine single-gpu, single-machine multi-gpu training. Multi-gpu dependence can be referred to [here](https://cg.cs.tsinghua.edu.cn/jittor/tutorial/2020-5-2-16-44-distributed/)
```shell
# Single GPU
bash train.sh

# Multiple GPUs
bash dist_train.sh
```

### Test
```shell
# Single GPU
bash test.sh

# Multiple GPUs
bash dist_test.sh
```
