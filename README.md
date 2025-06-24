# UR2P-Dehaze: Learning a Simple Image Dehaze Enhancer via Unpaired Rich Physical Prior

Abstract
===========================
Image dehazing techniques aim to enhance contrast and restore details, which are essential for preserving visual information and improving image processing accuracy. \textcolor{red}{Existing methods may struggle to capture the physical characteristics of images fully and deeply, which could limit their ability to reveal image details.} To overcome this limitation, we propose an unpaired image dehazing network, called the Simple Image Dehaze Enhancer via Unpaired Rich Physical Prior (UR2P-Dehaze). First, to accurately estimate the illumination, reflectance, and color information of the hazy image, we design a Shared Prior Estimator (SPE) that is iteratively trained to ensure the consistency of illumination and reflectance, generating clear, high-quality images. Additionally, a self-monitoring mechanism is introduced to eliminate undesirable features, providing reliable priors for image reconstruction. Next, we propose Dynamic Wavelet Separable Convolution (DWSC), which effectively integrates key features across both low and high frequencies, significantly enhancing the preservation of image details and ensuring global consistency. Finally, to effectively restore the color information of the image, we propose an Adaptive Color Corrector that addresses the problem of unclear colors. The PSNR, SSIM, LPIPS, FID and CIEDE2000 metrics on the benchmark dataset show that our method achieves state-of-the-art performance. It also contributes to the performance improvement of downstream tasks.

<img src="images/Frame.png" width="100%">

### News
[**2025.6.9**] The paper has been accepted by the journal Pattern Recognition. The final version includes additional experiments and content.  <br>
[**2025.1.17**] We have published **pre-trained weights** for different datasets. <br>
[**2025.1.13**] Publish the initial version, including code and dataset. <br>

Preparation
===========================
### Clone the repo

```sh
git clone https://github.com/Fan-pixel/UR2P-Dehaze.git
cd UR2P-Dehaze
```

## 🛠️ Install
Python 3.7 + Pytorch, please refer 'environment.yml' for detiled requirments.
You can create a new conda environment:
```
conda env create -f environment.yml
```
📦 Datasets
===========================
We used [SOTS-indoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0), [SOTS-outdoor](https://sites.google.com/view/reside-dehaze-datasets/reside-v0),[HSTS](https://sites.google.com/view/reside-dehaze-datasets/reside-v0)  and [I-HAZE](https://data.vision.ee.ethz.ch/cvl/ntire18//i-haze/) for testing.  

For training, we used [ITS](https://sites.google.com/view/reside-dehaze-datasets/reside-standard) dataset, you can follow the operations above to generate the training file lists.

## 💻 Training and Test
Training
You can modify the training settings for each experiment in the 'configs.yml'. Then run the following script to train the model：
```
CUDA_VISIBLE_DEVICES=xxxx python train.py --model （Model class） --checkpoints （Training sample address）
```
Testing
Run the following script to test the trained model：
```
CUDA_VISIBLE_DEVICES=XXX python test.py --model （Model class） --checkpoints （Test sample address）
```

Such as SOTS-indoor，SOTS-outdoor, you can download the pretrained models on [Training weight](https://pan.baidu.com/s/10OPzyosdEwYQoXcseZEliA)(uur1).

## 🚀 Qualitative Results
### Comparison Experiment
<img src="images/PSNRSSIM.png" width="80%">

### Dehazing results on SOTS-Indoor
<img src="images/indoor.png" width="80%">

### Dehazing results on SOTS-Outdoor
<img src="images/outdoor.png" width="80%">
<img src="images/outdoor1.png" width="80%">

### Dehazing results on HSTS
<img src="images/HSTS.png" width="80%">

### Dehazing results on I-HAZE
<img src="images/I-HAZE.png" width="80%">


## 🎓 Citation

If you find our work useful, please consider cite our paper:

```bibtex
@article{xue2025ur2p,
  title={UR2P-Dehaze: Learning a Simple Image Dehaze Enhancer via Unpaired Rich Physical Prior},
  author={Xue, Minglong and Fan, Shuaibin and Palaiahnakote, Shivakumara and Zhou, Mingliang},
  journal={arXiv preprint arXiv:2501.06818},
  year={2025}
}
```

## ✏️ Contact
If you have any questions, please contact the email Fansb@stu.cqut.edu.cn
