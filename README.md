<br />
<p align="center">
  <h1 align="center">Spiking Diffusion Models
    <br>
(IEEE Transactions on Artificial Intelligence)</h1>
  <p align="center" >
    Jiahang Cao<sup>1*</sup>,
    Hanzhong Guo<sup>2*</sup>,
    Ziqing Wang<sup>3*</sup>,
    Deming Zhou<sup>1</sup>,
    Hao Cheng<sup>1</sup>,
    Qiang Zhang<sup>1</sup>,
    Renjing Xu<sup>1†</sup>
<!--     <a href="https://evelinehong.github.io">Jiahang Cao*</a>,
    <a href="https://haoyuzhen.com">Ziqing Wang*</a>,
    <a href="https://peihaochen.github.io">Hanzhong Guo*</a>,
    <a href="https://zsh2000.github.io">Hao Cheng</a>,
    <a href="https://yilundu.github.io">Qiang Zhang</a>,
    <a href="https://zfchenunique.github.io">Renjing Xu</a> -->
  </p>
  <p align="center" >
    <em><sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)</em>
    <br>
    <em><sup>2</sup>The Hong Kong University</em> 
    <em><sup>3</sup>Northwestern University</em> 
  </p>
  <p align="center">
    <a href='https://github.com/AndyCao1125/SDM'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://github.com/AndyCao1125/SDM' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Proceeding-HTML-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Proceeding Supp'>
    </a>
  </p>
<!--   <p align="center">
    <img src="figs/illustration_main.png" alt="Logo" width="80%">
  </p> -->
</p>


This work **SDM** is an extended version of [**SDDPM**](https://github.com/AndyCao1125/SDDPM). We introduce several key improvements:

- **A New Family of Spiking-based Diffusion Models**：This work extends applicability to a wider array of diffusion solvers, including but not limited to DDPM, DDIM, Analytic-DPM and DPM-Solver.
- **Biologically Inspired Temporal-wise Spiking Mechanism (TSM)**: Inspired by biological processes that the neuron input at each moment experiences considerable fluctuations rather than being predominantly controlled by fixed synaptic weights, this module enables spiking neurons to capture more dynamic information. The TSM module can be integrated with existing modules (proposed by SDDPM) to further improve the image generation quality.
- **ANN-SNN Conversion for SDM**: To the best of our knowledge, we make the *first attempt* to utilize an ANN-SNN approach for implementing spiking diffusion models, complete with theoretical foundations.





## Acknowledgements & Contact
We thank the authors ([pytorch-ddpm](https://github.com/w86763777/pytorch-ddpm), [Fast-SNN](https://github.com/yangfan-hu/Fast-SNN), [spikingjelly](https://github.com/fangwei123456/spikingjelly)) for their open-sourced codes.

For any help or issues of this project, please contact jcao248@connect.hkust-gz.edu.cn.
