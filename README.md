# Pytorch-Segmentation-multi-models

Pytorch implementation for Semantic Segmentation with multi models (Deeplabv3, Deeplabv3_plus, PSPNet, UNet, UNet_AutoEncoder, UNet_nested, R2AttUNet, AttentionUNet, RecurrentUNet,, SEGNet, CENet, DsenseASPP, RefineNet, RDFNet) for blood vessel segmentation in fundus images of DRIVE dataset.

Data Available at https://www.isi.uu.nl/Research/Databases/DRIVE/

##Training
python train.py --model unet
The You can modify --model to change models.

##Reference:
AttentionR2Unet: Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation
https://arxiv.org/abs/1802.06955

AttentionUnet: Attention U-Net: Learning Where to Look for the Pancreas
https://arxiv.org/abs/1804.03999

CENet: CE-Net: Context encoder network for 2D medical image segmentation https://arxiv.org/abs/1903.02740

DeepLabV3:Rethinking Atrous Convolution for Semantic Image Segmentation（https://arxiv.org/pdf/1706.05587.pdf）

DeepLabV3_plus: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
https://arxiv.org/pdf/1802.02611.pdf

DenseASPP: DenseASPP for Semantic Segmentation in Street Scenes
http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf

PSPNet: Pyramid Scene Parsing Network
https://arxiv.org/abs/1612.01105

RDFNet: RDFNet: RGB-D Multi-level Residual Feature Fusion for Indoor Semantic Segmentation

RecurrentUnet: Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation
https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf

RefineNet: RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation
https://arxiv.org/pdf/1611.06612.pdf

SegNet: SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation
https://arxiv.org/abs/1511.00561

U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/abs/1505.04597

Unet_nested: Unet++: A Nested U-Net Architecture for Medical Image Segmentation
https://arxiv.org/pdf/1807.10165.pdf

##Github:
https://github.com/Guzaiwang/CE-N
https://github.com/ShawnBIT/UNet-family
https://github.com/charlesCXK/PyTorch_Semantic_Segmentation
