import numpy as np
import open3d as o3d
import util
import torch
import os
PC_PATH = '../self_sample_data/my_data/PU1K_raw_meshes/sampling/10000'
NOISE_PATH = '../'
pc_list = os.listdir(PC_PATH)
print(pc_list)
def add_gaussian_noise_to_point_cloud(point_cloud, mean=0, std_dev_position=0.01, std_dev_normal=0.5):
    # Convert point cloud data to NumPy arrays
    points = np.asarray(point_cloud.points)
    normals = np.asarray(point_cloud.normals)

    # Add Gaussian noise to positions
    print(points.shape)
    noise_position = np.random.normal(mean, std_dev_position, points.shape)
    noisy_points = points + noise_position

    # Add Gaussian noise to normals
    noise_normal = np.random.normal(mean, std_dev_normal, normals.shape)
    noisy_normals = normals + noise_normal

    # Create a new point cloud with noisy positions and normals
    noisy_point_cloud = o3d.geometry.PointCloud()
    noisy_point_cloud.points = o3d.utility.Vector3dVector(noisy_points)
    noisy_point_cloud.normals = o3d.utility.Vector3dVector(noisy_normals)
    # noisy_pc = torch.tensor(noisy_point_cloud) 
    # print(noisy_pc)
    return noisy_point_cloud

for pc_file in pc_list:
    if '.xyz' in pc_file:
        pc_file = '03046257.143e665cb61b96751311158f08f2982a.xyz'
        data_path = PC_PATH + '/' +pc_file
        save_path = NOISE_PATH + '/' + pc_file
        
        # Load the original point cloud
        original_point_cloud = o3d.io.read_point_cloud(data_path)

        # Add Gaussian noise to both positions and normals # 0.01
        noisy_point_cloud = add_gaussian_noise_to_point_cloud(original_point_cloud, std_dev_position=0.008, std_dev_normal=0.1)

        # Save the noisy point cloud
        o3d.io.write_point_cloud(save_path, noisy_point_cloud)
    break
# pc_file ='3642806.9fc5b76d363ca64ed03066fc8168e9c6.xyz'
# data_path = PC_PATH + '/' +pc_file
# save_path = NOISE_PATH + '/' + pc_file

# # Load the original point cloud
# original_point_cloud = o3d.io.read_point_cloud(data_path)

# # Add Gaussian noise to both positions and normals # 0.01
# noisy_point_cloud = add_gaussian_noise_to_point_cloud(original_point_cloud, std_dev_position=0.006, std_dev_normal=0.1)

# # Save the noisy point cloud
# o3d.io.write_point_cloud(save_path, noisy_point_cloud)
