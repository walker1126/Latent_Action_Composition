# LAC: Latent Action Composition for Skeleton-based Action Segmentation
### [Project Page](https://walker1126.github.io/LAC/) | [Paper](https://arxiv.org/pdf/2308.14500.pdf)
This is the official PyTorch implementation of the ICCV 2023 paper "LAC: Latent Action Composition for Skeleton-based Action Segmentation"

## Installation

Clone this repo:
```
git clone https://github.com/walker1126/Latent_Action_Composition.git
cd Latent_Action_Composition
```

Install dependencies:
```
pip install -r requirements.txt
```

## Action Generator (for Action Composition)

Generator training (by motion retargeting):
```
python train_generator.py -n view -g 01
```
Inference (action composition):
```
python predict.py -n view --model_path ./model/pretrained_view.pth -v1 ./examples/walk.json -v2 ./examples/drink.json -h1 720 -w1 720 -h2 720 -w2 720 -o ./outputs/com-demo --max_length 60
```

```bibtex
@InProceedings{Yang_2023_ICCV,
    author    = {Yang, Di and Wang, Yaohui and Dantcheva, Antitza and Kong, Quan and Garattoni, Lorenzo and Francesca, Gianpiero and Bremond, Francois},
    title     = {LAC - Latent Action Composition for Skeleton-based Action Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {13679-13690}
}
```
