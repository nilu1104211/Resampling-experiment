# https://github.com/fwilliams/point-cloud-utils#loading-meshes-and-point-clouds
# http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html
import open3d as o3d
import point_cloud_utils as pcu
import numpy as np
import math
import os
import pandas as pd
#option = ['1', '10', '100', '200']
option = ['1']
#MESH_PATH = '../self_sample_data/my_data/PU1K_raw_meshes/test/original_meshes/'
MESH_PATH = '../self_sample_data/my_data/PU1K_raw_meshes/test/original_meshes/'
GT_PATH = '../self_sample_data/my_data/PU1K_raw_meshes/sampling/39990/'
#GT_PATH = '../self_sample_data/my_data/PU1K_raw_meshes/test/input_2048/gt_8192/'
data_name = os.listdir(MESH_PATH) 
#data_name = ['02954340.40f0c6599f0be25fce01c07526cf2aa4', '03046257.143e665cb61b96751311158f08f2982a', '02828884.2b065fc9d62f1540ad5067eac75a07f7', '02691156.37f2f187a1582704a29fef5d2b2f3d7']
metrics = ['Mean_noise3', 'Mean_origin', '|Mean_diff|', 'CD_noise3', 'CD_origin', '|CD_diff|', 'CD(f)_noise3', 'CD(f)_origin', '|CD(f)_diff|', 'CD(b)_noise3', 'CD(b)_origin', '|CD(b)_diff|', 'HD_noise3', 'HD_origin', '|HD_diff|', 'proposed_win_cnt']
modes = ['curvature', 'density']
# for i in range(len(data_name)):
    # data_name[i] = data_name[i].split('.off')[0]

cur_df = pd.DataFrame(index=data_name, columns=metrics)
den_df = pd.DataFrame(index=data_name, columns=metrics)
for mode in modes:
    for opt in option:
        proposed_win_cnt = 0
        ORIGIN_PATH = '_result/pu1k_10k/origin/'
        PROPOSED_PATH = '_result/ablation/600/finetune/'
        for data in data_name:
            if "xyz" in data:
               continue
            data = data.split('.off')[0]
            # print(data)
            #print(MESH_PATH+data+'.off')
            #-------------------------- Mean -----------------------------------------
            mesh = o3d.io.read_triangle_mesh(MESH_PATH+data+'.off')
            # mesh = o3d.io.read_triangle_mesh(mesh_data)
            mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

            # Create a scene and add the triangle mesh
            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(mesh)  # we do not need the geometry ID for mesh
            # read point
            origin_point = o3d.io.read_point_cloud(ORIGIN_PATH+data+'/'+mode+'/result.xyz')
            proposed_point = o3d.io.read_point_cloud(PROPOSED_PATH+data+'/'+mode+'/result.xyz')
            origin_pts = np.asarray(origin_point.points)
            proposed_pts = np.asarray(proposed_point.points)
            origin_query_point = o3d.core.Tensor(origin_pts, dtype=o3d.core.Dtype.Float32)
            proposed_query_point = o3d.core.Tensor(proposed_pts, dtype=o3d.core.Dtype.Float32)

            # Compute distance of the query point from the surface
            ori_unsigned_distance = scene.compute_distance(origin_query_point)
            if len(ori_unsigned_distance)==0:
                continue
            ori_signed_distance = scene.compute_signed_distance(origin_query_point)
            ori_occupancy = scene.compute_occupancy(origin_query_point)

            pro_unsigned_distance = scene.compute_distance(proposed_query_point)
            if len(pro_unsigned_distance)==0:
                continue
            pro_signed_distance = scene.compute_signed_distance(proposed_query_point)
            pro_occupancy = scene.compute_occupancy(proposed_query_point)

            # origin_mean
            origin_mean = sum(ori_unsigned_distance)/len(ori_unsigned_distance)
            origin_mean = origin_mean.numpy()
            origin_mean = np.round(origin_mean, 4)
            # proposed mean
            proposed_mean = sum(pro_unsigned_distance)/len(pro_unsigned_distance)
            proposed_mean = proposed_mean.numpy()
            proposed_mean = np.round(proposed_mean, 4)
            # mean = str(mean).split('Tensor')[0]
            
            #rms = math.sqrt(sum(pow(unsigned_distance.numpy(),2))/len(unsigned_distance))
            #rms = np.round(rms, 7)
            
            #-------------------------- CD -----------------------------------------
            # CD forward # gt2reconstrcuted
            # origin
            gt_point = o3d.io.read_point_cloud(GT_PATH+data+'.xyz')
            ori_forward_dis = gt_point.compute_point_cloud_distance(origin_point)
            ori_backward_dis = origin_point.compute_point_cloud_distance(gt_point) 
            ori_f_distanceArray = np.asarray(ori_forward_dis)
            ori_b_distanceArray = np.asarray(ori_backward_dis)
            ori_cd_forward = np.mean(ori_f_distanceArray)
            ori_cd_backward = np.mean(ori_b_distanceArray)
            # proposed
            pro_forward_dis = gt_point.compute_point_cloud_distance(proposed_point)
            pro_backward_dis = proposed_point.compute_point_cloud_distance(gt_point) 
            pro_f_distanceArray = np.asarray(pro_forward_dis)
            pro_b_distanceArray = np.asarray(pro_backward_dis)
            pro_cd_forward = np.mean(pro_f_distanceArray)
            pro_cd_backward = np.mean(pro_b_distanceArray)
            # chamfer_distance
            # read point cloud
            # origin
            gt = np.asarray(gt_point.points)
            origin_re = np.asarray(origin_point.points)
            ori_chamfer_dist = pcu.chamfer_distance(gt, origin_re)
            # proposed
            pro_re = np.asarray(proposed_point.points)
            pro_chamfer_dist = pcu.chamfer_distance(gt, pro_re)

            #-------------------------- HD -----------------------------------------
            # Compute one-sided squared Hausdorff distances
            #hausdorff_a_to_b = pcu.one_sided_hausdorff_distance(gt, re)
            #hausdorff_b_to_a = pcu.one_sided_hausdorff_distance(re, gt)

            # Take a max of the one sided squared  distances to get the two sided Hausdorff distance
            ori_hausdorff_dist = pcu.hausdorff_distance(gt, origin_re)
            pro_hausdorff_dist = pcu.hausdorff_distance(gt, pro_re)

            # Find the index pairs of the two points with maximum shortest distancce
            #hausdorff_b_to_a, idx_b, idx_a = pcu.one_sided_hausdorff_distance(re, gt, return_index=True)
            #assert np.abs(np.sum((gt[idx_a] - re[idx_b])**2) - hausdorff_b_to_a**2) < 1e-5, "These values should be almost equal"
            # print("hausdorff_b_to_a")
            # print(hausdorff_b_to_a)
            # Find the index pairs of the two points with maximum shortest distancce
            #hausdorff_dist, idx_b, idx_a = pcu.hausdorff_distance(re, gt, return_index=True)
            #assert np.abs(np.sum((gt[idx_a] - re[idx_b])**2) - hausdorff_dist**2) < 1e-5, "These values should be almost equal"

            print(gt_point)
            print(origin_point)
            print(proposed_point)
            print('------')

            # df.at[data, 'Mean_'+opt] = mean
            # df.at[data, 'CD_'+opt] = chamfer_dist
            # df.at[data, 'HD_'+opt] = hausdorff_dist
            if origin_mean > proposed_mean:
                proposed_win_cnt+=1
            print(proposed_win_cnt)
            if mode == 'curvature':
                cur_df.at[data, 'Mean_origin'] = origin_mean
                cur_df.at[data, 'CD_origin'] = ori_chamfer_dist
                cur_df.at[data, 'CD(f)_origin'] = ori_cd_forward
                cur_df.at[data, 'CD(b)_origin'] = ori_cd_backward
                cur_df.at[data, 'HD_origin'] = ori_hausdorff_dist

                cur_df.at[data, 'Mean_noise3'] = proposed_mean
                cur_df.at[data, 'CD_noise3'] = pro_chamfer_dist
                cur_df.at[data, 'CD(f)_noise3'] = pro_cd_forward
                cur_df.at[data, 'CD(b)_noise3'] = pro_cd_backward
                cur_df.at[data, 'HD_noise3'] = pro_hausdorff_dist

                cur_df.at[data, '|Mean_diff|'] = abs(proposed_mean - origin_mean)
                cur_df.at[data, '|CD_diff|'] = abs(pro_chamfer_dist - ori_chamfer_dist)
                cur_df.at[data, '|CD(f)_diff|'] = abs(pro_cd_forward - ori_cd_forward)
                cur_df.at[data, '|CD(b)_diff|'] = abs(pro_cd_backward - ori_cd_backward)
                cur_df.at[data, '|HD_diff|'] = abs(pro_hausdorff_dist - ori_hausdorff_dist)
            else:
                den_df.at[data, 'Mean_origin'] = origin_mean
                den_df.at[data, 'CD_origin'] = ori_chamfer_dist
                den_df.at[data, 'CD(f)_origin'] = ori_cd_forward
                den_df.at[data, 'CD(b)_origin'] = ori_cd_backward
                den_df.at[data, 'HD_origin'] = ori_hausdorff_dist

                den_df.at[data, 'Mean_noise3'] = proposed_mean
                den_df.at[data, 'CD_noise3'] = pro_chamfer_dist
                den_df.at[data, 'CD(f)_noise3'] = pro_cd_forward
                den_df.at[data, 'CD(b)_noise3'] = pro_cd_backward
                den_df.at[data, 'HD_noise3'] = pro_hausdorff_dist

                den_df.at[data, '|Mean_diff|'] = abs(proposed_mean - origin_mean)
                den_df.at[data, '|CD_diff|'] = abs(pro_chamfer_dist - ori_chamfer_dist)
                den_df.at[data, '|CD(f)_diff|'] = abs(pro_cd_forward - ori_cd_forward)
                den_df.at[data, '|CD(b)_diff|'] = abs(pro_cd_backward - ori_cd_backward)
                den_df.at[data, '|HD_diff|'] = abs(pro_hausdorff_dist - ori_hausdorff_dist)
        if mode == 'curvature':
            cur_df.at[data, 'proposed_win_cnt'] = proposed_win_cnt
        else:
            den_df.at[data, 'proposed_win_cnt'] = proposed_win_cnt

# 엑셀 파일로 저장
file_path = 'multi_600.xlsx'
sheet_name = 'performence'

#cur_df.sort_values(by='|Mean_diff|', axis=0)
#den_df.sort_values(by='|Mean_diff|', axis=0)
cur_df.to_excel('cur'+file_path, sheet_name=sheet_name)
den_df.to_excel('den'+file_path, sheet_name=sheet_name)
print(proposed_win_cnt)
print('save!')