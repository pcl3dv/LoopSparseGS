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

# About LoopSparseGS

<div>
    LoopSparseGS is a loop-based 3DGS framework for the sparse novel view synthesis task. In specfic, we propose a loop-based 
    Progressive Gaussian Initialization (PGI) strategy that could iteratively densify the initialized point cloud using the rendered 
    pseudo images during the training process. Then, the sparse and reliable depth from the Structure from Motion, 
    and the window-based dense monocular depth are leveraged to provide precise geometric supervision via the proposed 
    Depth-alignment Regularization (DAR). Additionally, we introduce a novel Sparse-friendly Sampling (SFS) strategy to 
    handle oversized Gaussian ellipsoids leading to large pixel errors.
</div>

![image](assets/teaser.png)

<br>

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