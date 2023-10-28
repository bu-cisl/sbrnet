# SBR-Net
This repository contains the python implementations of the paper: **Robust single-shot 3D fluorescence imaging in scattering media with a simulator-trained neural network**. We provide a code in a python package, `sbrnet-core`, that contains functions for generating synthetic data and training models. 

### Citation
If you find this project useful in your research, please consider citing our paper:

[**J. Alido, J. Greene, Y. Xue, G. Hu, Y. Li, M. Gilmore, K. J. Monk, B. T. DeBenedicts, I. G. Davison, and L. Tian, "Robust single-shot 3D fluorescence imaging in scattering media with a simulator-trained neural network," (2023).**](https://arxiv.org/abs/2303.12573)

### Abstract
Imaging through scattering is a pervasive and difficult problem in many biological applications. The high background and the exponentially attenuated target signals due to scattering fundamentally limits the imaging depth of fluorescence microscopy. Light-field systems are favorable for high-speed volumetric imaging, but the 2D-to-3D reconstruction is fundamentally ill-posed and scattering exacerbates the condition of the inverse problem. Here, we develop a scattering simulator that models low-contrast target signals buried in heterogeneous strong background. We then train a deep neural network solely on synthetic data to descatter and reconstruct a 3D volume from a single-shot light-field measurement with low signal-to-background ratio (SBR). We apply this network to our previously developed Computational Miniature Mesoscope and demonstrate the robustness of our deep learning algorithm on a 75 micron thick fixed mouse brain section and on bulk scattering phantoms with different scattering conditions. The network can robustly reconstruct emitters in 3D with a 2D measurement of SBR as low as 1.05 and as deep as a scattering length. We analyze fundamental tradeoffs based on network design factors and out-of-distribution data that affect the deep learning model's generalizability to real experimental data. Broadly, we believe that our simulator-based deep learning approach can be applied to a wide range of imaging through scattering techniques where experimental paired training data is lacking.

### Data
All data to reproduce results from the paper can be found in this [**google drive**](https://drive.google.com/drive/folders/1XszF-KL4qUXUUmTcvLqYMwdYx6lP7dcG?usp=share_link).

### Usage
Clone the repo into a folder of your choice with <br>
`git clone https://github.com/bu-cisl/sbrnet.git` <br>
and set up the package with <br>
`python setup.py`. <br>
Ensure the necessary dependencies are installed using <br>
`pip install -r requirements.txt`.

## License
This project is licensed under the terms of the MIT license. see the [LICENSE](LICENSE) file for details
