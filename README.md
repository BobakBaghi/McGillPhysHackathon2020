# McGill Physics Hackathon 
This the LYF HAKS submission to the McGill 2020 physics hackathon. This repository was cloned and the information can be found below. All credit for the previous fork goes to the original authors. This repo was just modified to work with astronomical data.

# Self-supervised Deep Outlier Removal with Network Uncertainty and Score Refinement

By Siqi  Wang,  Yijie  Zeng,  Xinwang  Liu,  Sihang  Zhou,  En  Zhu,  Marius  Kloft,  Jianping  Yin,  Qing Liao.  Sumbitted to IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).

## Introduction

This repository provides a substantially improved version of the E3Outlier framework.  The codes are used to reproduce results reported in our manuscript submitted to TPAMI, which extends our original publication in NeurIPS 2019.  Some improvements are summarized as below:
- Multiple score refinement strategies (enseble, re-weighting) for E3Outlier are added. The refinements provide up to 2.1% AUROC gain on all benchmarks.  Check `outlier_experiments.py` for the implementations.
- Several recently published unsupervised outlier detection methods are added for comparisons, including DSEBM, RSRAE/RSRAE+, MO-GAAL.  Check `outlier_experiments.py` for the implementations.
- A script demonstrating the effect of uncertainty for a regression network can be found in `scripts/demo_uncertainty.py`.

The requirements and usages for this improved version are the same as the original E3Outlier implementation.  In specific, all the results of E3Outlier and other UOD methods can be obtained by running

```bash
python outlier_experiments.py
``` 

If you use this branch of E3Outlier for your research, please cite the following besides our original publication:

```
@article{wang_e3outlier_2020,
  title = {Self-supervised Deep Outlier Removal with Network Uncertainty and Score Refinement},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  author = {Wang, Siqi and Zeng, Yijie and Liu, Xinwang and Zhou, Sihang and Zhu, En and Kloft, Marius and Yin, Jianping and Liao, Qing},
  year = {2020},
  pubstate = {Under review},
}
```

-----

# Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network
By Siqi Wang, Yijie Zeng, Xinwang Liu, En Zhu, Jianping Yin, Chuanfu Xu, Marius Kloft.  In [NeurIPS 2019](https://papers.nips.cc/paper/8830-effective-end-to-end-unsupervised-outlier-detection-via-inlier-priority-of-discriminative-network).

## Introduction
This is the official implementation of the E3Outlier framework presented by "Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network".
The codes are used to reproduce experimental results of  E3Outlier and other unsupervised outlier detection (UOD) methods reported in the [paper](https://papers.nips.cc/paper/8830-effective-end-to-end-unsupervised-outlier-detection-via-inlier-priority-of-discriminative-network.pdf).

## Requirements
- Python 3.6
- PyTorch 0.4.1 (GPU)
- Keras 2.2.0 
- Tensorflow 1.8.0 (GPU)
- sklearn 0.19.1
 

## Usage

To obtain the results of E3Outlier and other UOD methods compared in the paper with default settings, simply run the following command:

```bash
python outlier_experiments.py
```

This will automatically run all UOD methods reported in the manuscript.  Please see ```outlier_experiments.py``` for more details.

After training, to print UOD results for a specific algorithm in AUROC/AUPR, run:

```bash
# AUROC of E3Outlier on CIFAR10 with outlier ratio 0.1
python evaluate_roc_auc.py --dataset cifar10 --algo_name e3outlier-0.1

# AUPR of CAE-IF on MNIST with outlier ratio 0.25 and inliers as the postive class
python evaluate_pr_auc.py --dataset mnist --algo_name cae-iforest-0.25 --postive inliers
```

The algorithm names are defined in ```outlier_experiments.py```.

## Citation

```
@incollection{NIPS2019_8830,
title = {Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network},
author = {Wang, Siqi and Zeng, Yijie and Liu, Xinwang and Zhu, En and Yin, Jianping and Xu, Chuanfu and Kloft, Marius},
booktitle = {Advances in Neural Information Processing Systems 32},
editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
pages = {5960--5973},
year = {2019},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/8830-effective-end-to-end-unsupervised-outlier-detection-via-inlier-priority-of-discriminative-network.pdf}
}
```

## License

E3Outlier is released under the MIT License.
