import math
import torch
import random
import util
from torch.utils.data import Dataset
from pathlib import Path
import os
import open3d as o3d
import numpy as np
from typing import List
from kmeans_pytorch import kmeans
NUM_CLUSTERS = 5


def get_multi_dataset(mode='sweep'):
    mode = mode.lower().strip()

    if mode == 'sweep':
        return SweepData

    if mode == 'random':
        return Random

    if mode == 'curvature':
        return CurvatureData

    if mode == 'density':
        return DensityData

    if mode == 'saliency':
        return PointFeatureData

    if mode == 'noise':
        return NoiseData
    
    if mode == 'pretrain':
        return PretrainPerturbationData
    
    if mode == 'denoising':
        return Denoising

class SweepData(Dataset):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        self.const = None

        # for sweep
        self.reminder = -1
        self.blocks = int(self.pc.shape[0] / (args.D2 + args.D1))

    def single(self, size):
        return self.pc[torch.randperm(self.n_pts)[:size], :3].permute(1, 0).unsqueeze(0).to(self.real_device)

    def __getitem__(self, item):
        current_block = int(item / self.blocks)
        if self.reminder != current_block:
            # shuffle pc
            self.pc = self.pc[torch.randperm(self.pc.shape[0]), :]
            self.reminder = current_block

        index = item % self.blocks
        offset = index * (self.args.D1 + self.args.D2)
        d1 = self.pc[offset: offset + self.args.D1, :3].transpose(0, 1)
        d2 = self.pc[offset + self.args.D1: offset + self.args.D1 + self.args.D2,
             :3].transpose(0, 1)

        return d1, d2

    def __len__(self):
        return self.args.iterations

class Random(Dataset):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

    def single(self, size):
        return self.pc[torch.randperm(self.n_pts)[:size], :3].permute(1, 0)

    def __getitem__(self, item):
        return self.single(self.args.D1).to(self.device), self.single(self.args.D2).to(self.device)

    def __len__(self):
        return self.args.iterations

class SubsetData(Dataset):
    def __init__(self, noise_pc, clean_pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.noise_pc: torch.Tensor = noise_pc.to(self.device)
        self.clean_pc: torch.Tensor = clean_pc.to(self.device)
        self.n_pts = self.clean_pc.shape[0]

        # if self.args.kmeans:
        #     cluster_indx, cluster_centers = kmeans(
        #         X= weight.unsqueeze(-1), num_clusters=NUM_CLUSTERS, distance='euclidean', device=real_device
        #     )
        #     min_indx = torch.argmin(cluster_centers)
        #     self.criterion_mask = (cluster_indx == min_indx)

        # elif self.args.percentile == -1.0:
        #     self.criterion_mask = weight < weight.mean().to(self.device)
        # else:
        #     kth = weight.kthvalue(int(weight.shape[0] * self.args.percentile))[0]
        #     self.criterion_mask = weight < kth

        # self.high_pc = pc[self.criterion_mask, :].to(self.device)
        # self.low_pc = pc[~self.criterion_mask, :].to(self.device)
        self.high_pc = self.clean_pc # 원하는 모양
        self.low_pc = self.noise_pc

        # self.export_marked()

        # subset ratios
        self.p1 = self.args.p1
        self.p2 = self.args.p2

        self.const = None

    def single(self, size):
        return self.pc[torch.randperm(self.n_pts)[:size], :3].permute(1, 0).unsqueeze(0).to(self.real_device)

    def __getitem__(self, item):
        size1, size2 = self.args.D1, self.args.D2
        D1, D2 = SubsetData.double_sub_sample(self.high_pc, self.low_pc, self.p1,
                                                 self.p2, size1, size2,
                                                 allow_residual=True)

        return D1[:, :3].permute(1, 0), D2[:, :3].permute(1, 0)

    def __len__(self):
        return self.args.iterations

    @staticmethod
    def double_sub_sample(pc1: torch.Tensor, pc2: torch.Tensor, p1, p2, n1, n2, allow_residual=True):
        high_perm = torch.randperm(pc1.shape[0])
        low_perm = torch.randperm(pc2.shape[0])

        d1np1 = int(n1 * p1)
        d1np2 = int(n1 * (1 - p1))
        if d1np1 > pc1.shape[0]:
            d1np2 += (d1np1 - pc1.shape[0])
            d1np1 = pc1.shape[0]
        if d1np2 > pc2.shape[0]:
            d1np1 += (d1np2 - pc2.shape[0])
            d1np2 = pc2.shape[0]

        d2np1 = int(n2 * p2)
        d2np2 = int(n2 * (1 - p2))
        if d2np1 > pc1.shape[0]:
            d2np2 += (d2np1 - pc1.shape[0])
            d2np1 = pc1.shape[0]
        if d2np2 > pc2.shape[0]:
            d2np1 += (d2np2 - pc2.shape[0])
            d2np2 = pc2.shape[0]

        d1idx1, d2idx1 = SubsetData.disjoint_select(high_perm, d1np1, d2np1, allow_residue=allow_residual)

        d2np2 += max(0, d2np1 - d2idx1.shape[0])

        d1idx2, d2idx2 = SubsetData.disjoint_select(low_perm, d1np2, d2np2, allow_residue=allow_residual)

        return torch.cat([pc1[d1idx1, :], pc2[d1idx2, :]], dim=0), torch.cat([pc1[d2idx1, :], pc2[d2idx2, :]],
                                                                             dim=0)

    @staticmethod
    def disjoint_select(pc, n1, n2, allow_residue=True):
        idx1 = pc[:n1]
        if allow_residue:
            residual = max(n1 + n2 - pc.shape[0], 0)
        else:
            residual = 0
        idx2 = pc[max(n1 - residual, 0): n1 + n2]
        return idx1, idx2


class CurvatureData(SubsetData):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]
        if args.curvature_cache and os.path.exists(args.curvature_cache):
            self.curvature = torch.load(args.curvature_cache)
            print('loaded curvature metric from cache')
        else:
            self.curvature: torch.Tensor = self.get_curvature().to(self.device)
            if args.curvature_cache:
                torch.save(self.curvature, args.curvature_cache)

        # lowest values are selected for subset division
        self.curvature *= -1
        super().__init__(pc, self.curvature, real_device, args)

        print(f'Sharp Shape: {self.high_pc.shape}; Low Shape {self.low_pc.shape}')

    def get_curvature(self):
        div = util.angle_diff(self.pc, self.args.k)
        div[div != div] = 0
        return div


class DensityData(SubsetData):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        self.density: torch.Tensor = util.density(pc, args.k).to(
            self.device)

        super().__init__(pc, torch.log(self.density), real_device, args)

        print(f'Not Dense Shape: {self.high_pc.shape}; Dense Shape {self.low_pc.shape}')

class NoiseData(SubsetData):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        self.pc, self.noise_ind = self.preprocess_pc(self.pc)
        self.pc = self.pc.to(self.device)

        super().__init__(self.pc, self.noise_ind, real_device, args) # Since noise was added to self.pc, the modified points must be used!

        print(f'Noisy Shape(Pos): {self.high_pc.shape}; Noisy Shape(Neg) {self.low_pc.shape}')

    def preprocess_pc(self, pc):
        # convert tensor(n, 6) -> numpy(n, 3)-> o3d point cloud
        numpy_pc = pc.numpy()[:,:3] # (n, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(numpy_pc)

        diag = pcd.get_max_bound() - pcd.get_min_bound()
        diag_len = math.sqrt(diag[0] * diag[0] + diag[1] * diag[1] + diag[2] * diag[2])

        source_sample_ind = torch.randperm(self.n_pts)[:int(self.n_pts * self.args.noise_ratio)]
        pc[source_sample_ind,:3] += pc[source_sample_ind, 3:6] * torch.rand(len(source_sample_ind), 1) * (self.args.noise_level * diag_len)

        noise_ind = torch.zeros(self.n_pts)
        noise_ind[source_sample_ind] = 1

        return torch.Tensor(pc), noise_ind


class PointFeatureData(SubsetData):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        self.saliency = self.preprocess_pc(self.pc).to(self.device)
        self.saliency *= -1 # (TODO: check)

        super().__init__(pc, self.saliency, real_device, args)

        print(f'Salient Shape: {self.high_pc.shape}; Not Salient Shape {self.low_pc.shape}')

    def preprocess_pc(self, pc):
        # convert tensor(n, 6) -> numpy(n, 3)-> o3d point cloud
        numpy_pc = pc.numpy()[:,:3] # (n, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(numpy_pc)

        diag = pcd.get_max_bound() - pcd.get_min_bound()
        diag_len = math.sqrt(diag[0] * diag[0] + diag[1] * diag[1] + diag[2] * diag[2])

        # estimate normal
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=diag_len*0.01, max_nn=self.args.k))

        # calculate FPFH feature(http://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html?highlight=fpfh)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(radius=diag_len*0.02, max_nn=self.args.k))

        numpy_fpfh = np.transpose(np.asarray(pcd_fpfh.data)) # (n, 33)
        mean_fpfh = np.mean(numpy_fpfh, axis=0) # (33)
        distance_from_mean = np.linalg.norm(numpy_fpfh - mean_fpfh,axis=1) # (n) use as a weight

        if self.args.inverted == False:
            return torch.Tensor(distance_from_mean)
        else:
            return torch.Tensor(-distance_from_mean)
        
# Pretrain용 데이터 생성기
class PretrainPerturbationData(Dataset):
    def __init__(self, pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.pc: torch.Tensor = pc.to(self.device)
        self.n_pts = self.pc.shape[0]

        self.pc, self.noise_ind = self.preprocess_pc(self.pc)
        self.pc = self.pc.to(self.device)
        self.noise_ind = self.noise_ind.to(self.device)

    def preprocess_pc(self, pc):
        # convert tensor(n, 6) -> numpy(n, 3)-> o3d point cloud
        numpy_pc = pc.numpy()[:,:3] # (n, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(numpy_pc)

        diag = pcd.get_max_bound() - pcd.get_min_bound()
        diag_len = math.sqrt(diag[0] * diag[0] + diag[1] * diag[1] + diag[2] * diag[2])

        # TODO: 
        # For now, selected 0.2 noise ratio and 0.016 noise ratio based on the PCQA paper. 
        # If necessary, these should be added as options later.
        source_sample_ind = torch.randperm(self.n_pts)[:int(self.n_pts * 0.2)]
        pc[source_sample_ind,:3] += torch.rand(len(source_sample_ind), 1) * (0.016 * diag_len)

        noise_mask = torch.zeros(self.n_pts)
        noise_mask[source_sample_ind] = 1

        # noise_ind is a mask where 1 indicates noise and 0 indicates non-noise
        return torch.Tensor(pc), noise_mask

    def single(self, size):
        return self.pc[torch.randperm(self.n_pts)[:size], :3].permute(1, 0).unsqueeze(0).to(self.real_device)

    def __getitem__(self, item):
        # In pretraining as well, a point cloud of size D1 is returned at each iteration
        source_sample_ind = torch.randperm(self.n_pts)[:self.args.D1]
        return self.pc[source_sample_ind,:3].transpose(0,1), self.noise_ind[source_sample_ind]

    def __len__(self):
        return self.args.iterations
    
          
class Denoising(SubsetData):
    def __init__(self, noise_pc, clean_pc, real_device, args):
        self.device = torch.device('cpu')
        self.real_device = real_device
        self.args = args
        self.noise_pc: torch.Tensor = noise_pc.to(self.device)
        self.clean_pc: torch.Tensor = clean_pc.to(self.device)
        self.n_pts = self.clean_pc.shape[0]

        super().__init__(self.noise_pc, self.clean_pc, real_device, args) # Since noise has been added to self.pc, the modified points must be used!

        print(f'Noisy Shape(Pos): {self.high_pc.shape}; Noisy Shape(Neg) {self.low_pc.shape}')
