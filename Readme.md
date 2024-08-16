## Universal Style Transfer

This is the Pytorch implementation of [Universal Style Transfer via Feature Transforms](https://arxiv.org/pdf/1705.08086.pdf).

Official Torch implementation can be found [here](https://github.com/Yijunmaverick/UniversalStyleTransfer) and Tensorflow implementation can be found [here](https://github.com/eridgd/WCT-TF).

## Prerequisites
- [Pytorch](http://pytorch.org/)
- [torchvision](https://github.com/pytorch/vision)
- Pretrained encoder and decoder [models](https://drive.google.com/file/d/0B8_MZ8a8aoSeWm9HSTdXNE9Eejg/view?usp=sharing&resourcekey=0-MokQLSx3g7mqN50Eli5G4w) for image reconstruction only (download and uncompress them under models/)
- CUDA + CuDNN

## Environment
- Pytorch: 1.13.1
- Torchvision: 0.14.1
- CUDA 11.7

## Prepare images
Simply put content and image pairs in `images/content` and `images/style` respectively. Note that correspoding conternt and image pairs should have same names.


## Style Transfer

```
python WCT.py --cuda
```

## Results
<img src="images/content/in1.jpg" width="200" hspace="5"><img src="images/style/in1.jpg" width="200" hspace="5"><img src="images/content/in3.jpg" width="200" hspace="5"><img src="images/style/in3.jpg" width="200" hspace="5">

<img src="samples/in1.jpg" width="400" hspace="10"><img src="samples/in3.jpg" width="400" hspace="10">

<img src="images/content/in2.jpg" width="200" hspace="5"><img src="images/style/in2.jpg" width="200" hspace="5"><img src="images/content/in4.jpg" width="200" hspace="5"><img src="images/style/in4.jpg" width="200" hspace="5">

<img src="samples/in2.jpg" width="400" hspace="10"><img src="samples/in4.jpg" width="400" hspace="10">

### Acknowledgments
Many thanks to the author Yijun Li for his kind help.

### Reference
Li Y, Fang C, Yang J, et al. Universal Style Transfer via Feature Transforms[J]. arXiv preprint arXiv:1705.08086, 2017.
