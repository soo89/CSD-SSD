# [CSD: Consistency-based Semi-supervised learning for object Detection](https://papers.nips.cc/paper/9259-consistency-based-semi-supervised-learning-for-object-detection)

By [Jisoo Jeong](http://mipal.snu.ac.kr/index.php/Jisoo_Jeong), Seungeui Lee, [Jee-soo Kim](http://mipal.snu.ac.kr/index.php/Jee-soo_Kim), [Nojun Kwak](http://mipal.snu.ac.kr/index.php/Nojun_Kwak)

## Installation & Preparation
We experimented with CSD using the SSD pytorch framework. To use our model, complete the installation & preparation on the [SSD pytorch homepage](https://github.com/amdegroot/ssd.pytorch)

#### prerequisites
- Python 3.6
- Pytorch 1.0.0

## Supervised learning
```Shell
python train_ssd.py
```

## CSD training
```Shell
python train_csd.py
```

## Evaluation
```Shell
python eval.py
```
