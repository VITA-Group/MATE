# MATE: Plugging in Model Awareness to Task Embedding for Meta Learning

This repository contains the code for the paper:

MATE: Plugging in Model Awareness to Task Embedding for Meta Learning

[Xiaohan Chen](http://xiaohanchen.com), Zhangyang Wang,
[Siyu Tang](https://ps.is.mpg.de/person/stang), [Krikamol Muandet](http://www.krikamol.org/)

This paper is accepted in NeurIPS 2020. The link to the proceedings will be available soon.

## Abstract

Meta-learning improves generalization of machine learning models when faced with previously unseen tasks by leveraging experiences from different, yet related prior tasks. To allow for better generalization, we propose a novel task representation called model-aware task embedding (MATE) that incorporates not only the data distributions of different tasks, but also the complexity of the tasks through the models used. The task complexity is taken into account by a novel variant of kernel mean embedding, combined with an instance-adaptive attention mechanism inspired by an SVM-based feature selection algorithm. Together with conditioning layers in deep neural networks, MATE can be easily incorporated into existing meta learners as a plug-and-play module. While MATE is widely applicable to general tasks where the concept of task/environment is involved, we demonstrate its effectiveness in few-shot learning by improving a state-of-the-art model consistently on two benchmarks.

<!-- ### Citation

If you use this code for your research, please cite our paper:
```
@inproceedings{,
  title={MATE: Plugging in Model Awareness to Task Embedding for Meta Learning},
  author={Xiaohan Chen and Zhangyang Wang and Siyu Tang and Krikamol Muandet},
  booktitle={NeurIPS},
  year={2020}
}
``` -->

## Dependencies

* Python 2.7 (not tested on Python 3)
* [PyTorch 0.4.1](http://pytorch.org)
* [qpth 0.0.11+](https://github.com/locuslab/qpth)
* [tqdm](https://github.com/tqdm/tqdm)

## Usage

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/kjunelee/MetaOptNet.git
    cd MetaOptNet
    ```

2. Datasets. Please check the official repo of MetaOptNet ([link](https://github.com/kjunelee/MetaOptNet))
   for the processed dataset data.

3. For each dataset loader, configure the path to the directory in `data/your_dataset.py`.
   For example, in `data/mini_imagenet.py`:

   ```python
   _MINI_IMAGENET_DATASET_DIR = 'path/to/miniImageNet'
   ```

### Meta-training

Take CIFAR-FS dataset for example of how to run the codes.
The training process of MATE consists of two steps:

1. First, pre-train a feature extractor with using any method you'd like, e.g.
    using MetaOptNet with a SVM head:

    ```bash
    python train.py --gpu 0 --save-path "./experiments/CIFAR_FS_MetaOptNet_SVM" --train-shot 5 \
    --head SVM --network ResNet --dataset CIFAR_FS
    ```

1. Train MATE using the pre-trained ResNet12 feature extractor with DualBN and
   MSGAN regularization on the FiLM modules.

    ```bash
    python train_film.py --gpu 0,1 \
        --save-path ./experiments/CIFAR_FS_MATE_SVM \
        --save-epoch 10 --train-shot 5 --dataset CIFAR_FS --val-episodes-per-batch 4 \
        --head SVM --network ResNet_FiLM \
        --task-embedding FiLM_SVM_WGrad --dual-BN --lr 0.01 \
        --film-reg-type MSGAN --film-reg-level 1e-6 \
        --load-naive-backbone --load path/to/the/pre-trained/model
    ```

### Meta-testing

1. To test MATE-SVM on 5-way CIFAR-FS 1-shot benchmark:

    ```bash
    python test_film.py --gpu 0 --episode 1000 --way 5 --shot 1 --query 15 \
        --dataset CIFAR_FS \
        --head SVM --network ResNet_FiLM \
        --task-embedding FiLM_SVM_WGrad --dual-BN \
        --load path/to/the/tested/model
    ```

1. Similarly, to test MATE-SVM on 5-way CIFAR-FS 5-shot benchmark:

    ```bash
    python test_film.py --gpu 0 --episode 1000 --way 5 --shot 5 --query 15 \
        --dataset CIFAR_FS \
        --head SVM --network ResNet_FiLM \
        --task-embedding FiLM_SVM_WGrad --dual-BN \
        --load $load_path
    ```

## Acknowledgments

This code is based on the implementations of [MetaOptNet](https://github.com/kjunelee/MetaOptNet).
