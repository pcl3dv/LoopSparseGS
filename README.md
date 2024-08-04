<div align="center">

<h1>LoopSparseGS: Loop Based Sparse-View Friendly Gaussian Splatting </h1>

<div>
    Zhenyu Bao<sup>1</sup>&emsp;
    Guibiao Liao<sup>1, *</sup>&emsp;
    Kaichen Zhou<sup>1</sup>&emsp;
    Kanglin Liu<sup>2</sup>&emsp;
    Qing Li<sup>2, *</sup>&emsp;
    Guoping Qiu<sup>3</sup>
</div>

<div>
    <sup>1</sup>Peking University&emsp;
    <sup>2</sup>Pengcheng Laboratory&emsp;
    <sup>3</sup>University of Nottingham
</div>

<div>
    <sup>*</sup>corresponding author
</div>

### [Paper](https://arxiv.org/abs/2408.00254) | [Project](https://zhenybao.github.io/LoopSparseGS) | Video | Code ( is coming soon... )

</div>

![image](assets/comparison.gif)

<br>

# 3D Gaussian Splatting in Sparse Setting
<div>
    Despite the photorealistic novel view synthesis (NVS) performance achieved by the original 3D Gaussian splatting (3DGS), 
    its rendering quality significantly degrades with sparse input views. This performance drop is mainly caused by several challenges. 
    Firstly, given the sparse input views, the initial Gaussian points provided by Structure from Motion (SfM) can be sparse and inadequate, 
    as shown in follow figure (top left). 
    Secondly, reconstructing the appearance and geometry of scenes becomes an under-constrained and ill-posed issue with insufficient inputs with only the image reconstruction constraints. 
    Thirdly, the scales of some Gaussians grow to be very large during the optimization process, 
    and these oversized Gaussian ellipsoids result in the overfitting problem, thus producing unsatisfactory results at novel viewpoints as illustrated in follow figure (top middle).
</div>

<br>

![image](assets/teaser.png)

# LoopSparseGS Method
<div>
    LoopSparseGS is a loop-based 3DGS framework for the sparse novel view synthesis task. In specfic, we propose a loop-based 
    Progressive Gaussian Initialization (PGI) strategy that could iteratively densify the initialized point cloud using the rendered 
    pseudo images during the training process. Then, the sparse and reliable depth from the Structure from Motion, 
    and the window-based dense monocular depth are leveraged to provide precise geometric supervision via the proposed 
    Depth-alignment Regularization (DAR). Additionally, we introduce a novel Sparse-friendly Sampling (SFS) strategy to 
    handle oversized Gaussian ellipsoids leading to large pixel errors.
</div>

<br>

![image](assets/main.png)



# Quantitative comparison

![image](assets/metrix1.png)



![image](assets/metrix2.png)



# Qualitative comparison

![image](assets/visual1.png)



![image](assets/visual2.png)



## Citation

Cite as below if you find this repository is helpful to your project:
```
@article{bao2024loopsparsegs,
      title={LoopSparseGS: Loop Based Sparse-View Friendly Gaussian Splatting},
      author={Bao, Zhenyu and Liao, Guibiao and Zhou, Kaichen and Liu, Kanglin and Li, Qing and Qiu, Guoping},
      journal={arXiv preprint arXiv:2408.00254},
      year={2024},
    }
```