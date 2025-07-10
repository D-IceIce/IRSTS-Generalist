# IRSTS Generalist: Improving Generalization in Infrared Small Target Segmentation Using One Shot

This repository contains the implementation of the paper ''**IRSTS Generalist: Improving Generalization in Infrared Small Target Segmentation Using One Shot**''.

## Dataset Setup

First, download the [IRDST](http://xzbai.buaa.edu.cn/datasets.html) dataset and place it in the `IRDST` folder with the following structure:

```
IRDST/
└── real/
├── images/
└── masks/
├── 1/
│   ├── 1(1).png
│   ├── 1(2).png
│   └── ...
├── 2/
│   ├── 1(1).png
│   ├── 1(2).png
│   └── ...
├── 3/
│   └── ...
```


## Baseline & Foundation Model Setup

Next, download the **baseline** and **foundation** models and place them into the `baseline/` and `foundation/` folders respectively:

```
baseline/
├── DNANet/
├── MSHNet/
└── ...
foundation/
├── DINO/
├── SAM/
└── ...
````
Download link: 
[DNANet](https://github.com/YeRen123455/Infrared-Small-Target-Detection) |
[MSHNet](https://github.com/Lliu666/MSHNet) |
[DINO](https://github.com/facebookresearch/dino) |
[SAM](https://github.com/facebookresearch/segment-anything)


## Running the Code

Finally, install the necessary libraries and run the code in the following order:

* Run `oneshot_train.py` to train the classifier
* Run `oneshot_baseline_test.py` to test the baseline model
* Run `oneshot_foundation_test.py` to test the foundation model.