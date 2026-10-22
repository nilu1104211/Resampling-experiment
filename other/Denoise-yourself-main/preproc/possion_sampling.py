import open3d as o3d
import os

MESH_PATH = 'my_data/PU1K_raw_meshes/test/original_meshes/'
POINT_PATH = 'my_data/PU1K_raw_meshes/sampling/39990/'
mesh_list = os.listdir(MESH_PATH)
# print(len(mesh_list)) # 100
for i, mesh in enumerate(mesh_list):
    print(i)
    name = mesh.split('.off')[0]
    # print(MESH_PATH + mesh)
    # print(POINT_PATH + name + '.xyz')
    m_path = MESH_PATH + mesh
    p_path = POINT_PATH + name + '.xyz'
    # read mesh file
    mesh = o3d.io.read_triangle_mesh(m_path) 

    # mesh to point # possion disk sampling
    point_cloud = mesh.sample_points_poisson_disk(39990)

    # write point cloud
    o3d.io.write_point_cloud(p_path, point_cloud)