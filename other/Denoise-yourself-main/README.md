# Denoise yourself: Self-supervised point cloud upsampling with pretrained denoising
Offcial implementation of ["Denoise yourself: Self-supervised point cloud upsampling with pretrained denoising"](https://www.sciencedirect.com/science/article/pii/S095741742500260X).   
The paper has been accepted by Expert Systems with Applications (ESWA) 2025 (Q1 SCIE)

## Abstract
In this study, we propose a novel self-supervised approach for point cloud upsampling, integrating a pretrained denoising phase to enhance the quality and accuracy of the resulting upsampled point clouds. Most point cloud upsampling methods rely on supervised learning, requiring extensive datasets and complex parameter tuning. In contrast, our approach leverages self-sampling techniques to minimize the need for large labeled datasets, while addressing inherent noise issues through a dedicated denoise pretrain phase. We evaluate the performance of our method using the PU1K dataset, demonstrating significant improvements in the reduction of noise and the preservation of geometric features compared to baseline methods. Our proposed multi-object pretrain method outperforms existing methods, especially when using a relatively large number of points, across all performance metrics in curvature and density consolidation strategies. Additionally, our ablation study confirms that the multi-object pretrain method achieves superior performance with fewer fine-tuning iterations than traditional methods. Our experiments indicate that the proposed method effectively balances the trade-offs between data efficiency and upsampling quality, making it a robust solution for various 3D applications.

Authors: Ji-Hyeon Hur, Soonjo Kwon, Hyungki Kim

![structure](https://github.com/user-attachments/assets/00bd92e4-3abb-48a6-bce5-c61f8aab36c9)

![architecture](https://github.com/user-attachments/assets/52536401-74de-4805-b732-90fda2e7e0fc)


## Preparation
1. Clone the repository
```
git clone https://github.com/qhur135/Denoise-yourself.git
cd Denoise-yourself
```
2. Set up the virtual environment
```
conda env create -f env.yml
conda activate denoise-yourself
```
3. Download the dataset
Download PU1K dataset from [PU-GCN repository](https://github.com/guochengqian/PU-GCN?tab=readme-ov-file)    
Create a new folder named `my_data` inside the `Denoise-yourself` directory and store the downloaded dataset there.   

4. Sample 3D mesh datasets into point clouds
Use the script `Denoise-yourself/preproc/poisson_sampling.py` to perform sampling.   
`MESH_PATH` should point to the directory containing the input meshes, and `POINT_PATH` should specify the directory where the sampled point clouds will be saved.    
Pass the desired number of points as an argument to `sample_points_poisson_disk`, as shown below:  
```
  point_cloud = mesh.sample_points_poisson_disk(39990)
```

## Experiments 

We provide bash scripts to run different experimental settings.   

1. baseline model
```
bash _demos/10k_origin.sh  # for input point clouds with 10k points
bash _demos/2048_origin.sh # for input point clouds with 2048 points
```

2. denoise model
```
bash _demos/10k_denoise.sh  # for input point clouds with 10k points
bash _demos/2048_denoise.sh # for input point clouds with 2048 points
```

3. multi model
   
3-1. pretrain    
```
bash _demos/10k_multi/10k_multi_pretrain.sh  # for input point clouds with 10k points
bash _demos/2048_multi/2048_multi_pretrain.sh # for input point clouds with 2048 points
```
3-2. finetune   
```
bash _demos/10k_multi/10k_multi_finetune.sh  # for input point clouds with 10k points
bash _demos/2048_multi/2048_multi_finetune.sh # for input point clouds with 2048 points
```

### Notes
You can set the `mode` variable to either `density` or `curvature` inside the scripts.    
`data_dir` specifies the path to the input point clouds.
`result_dir` specifies the path where the upsampled point clouds will be saved.   
In the pretrain stage, upsampling is not performed. Instead, the trained model weights will be saved.

### Citation
```bibtex
@article{hur2025denoise,
  title={Denoise yourself: Self-supervised point cloud upsampling with pretrained denoising},
  author={Hur, Ji-Hyeon and Kwon, Soonjo and Kim, Hyungki},
  journal={Expert Systems with Applications},
  volume={271},
  pages={126638},
  year={2025},
  publisher={Elsevier}
}


