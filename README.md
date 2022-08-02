
# WhiteNNer-Blind Image Denoising via Noise Whiteness Priors

![alt text](https://github.com/saeedizadi/whitenner/raw/main/figure.png?raw=true)

# Abstract

The accuracy of medical imaging-based diagnostics is directly impacted by the quality of the collected images. A passive approach to improve image quality is one that lags behind improvements in imaging hardware, awaiting better sensor technology of acquisition devices. An alternative, active strategy is to utilize prior knowledge of the imaging system to directly post-process and improve the acquired images. Traditionally, priors about the image properties are taken into account to restrict the solution space. However, few techniques exploit the prior about the noise properties. In this paper, we propose a neural network-based model for disentangling the signal and noise components of an input noisy image, without the need for any ground truth training data. We design a unified loss function that encodes priors about signal as well as noise estimate in the form of regularization terms. Specifically, by using total variation and piecewise constancy priors along with noise whiteness priors such as auto-correlation and stationary losses, our network learns to decouple an input noisy image into the underlying signal and noise components. We compare our proposed method to Noise2Noise and Noise2Self, as well as non-local mean and BM3D, on three public confocal laser endomicroscopy datasets. Experimental results demonstrate the superiority of our network compared to state-of-the-art in terms of PSNR and SSIM.

# Keywords
Confocal Laser Endomicroscopy, Denoising, Deep learning

# Cite
If you use our code, please cite our paper: 
[WhiteNNer-Blind Image Denoising via Noise Whiteness Priors?](https://ieeexplore.ieee.org/abstract/document/9022105)

[PDF]{https://www2.cs.sfu.ca/~hamarneh/ecopy/iccv_vrmi2019b.pdf}

The corresponding bibtex entry is:

```
@inproceedings{izadi2019whitenner,
  title={Whitenner-blind image denoising via noise whiteness priors},
  author={Izadi, Saeed and Mirikharaji, Zahra and Zhao, Mengliu and Hamarneh, Ghassan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision - Workshop on Visual Recognition of Medical Imaging},
  pages={476--484},
  year={2019}
}
```
# Usage
An example usage is shown in `demo.ipynb` The developed loss terms are calculated for a sample network output given a noisy image. 
