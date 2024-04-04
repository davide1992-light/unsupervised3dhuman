import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import open3d as o3d
import trimesh

file = "./results/ex1front_pred.ply"

pcd = o3d.io.read_point_cloud(file)
xyz = np.array(pcd.points)

n = 1
file = f"./results/ex{n}front_pred.ply"

# Load the PLY file
mesh = trimesh.load_mesh(file)
xyz = np.array(mesh.vertices)

left_arm = [627, 628, 789, 1232, 1233, 1311, 1315, 1378, 1379, 1381, 1382, 1385, 1388, 1389, 1393, 1394, 1396, 1397]


h = -0.015
plane_top = o3d.geometry.TriangleMesh()
np_vertices = np.array(
    [[h, 0.5, 0.5], [h, -0.5,- 0.5], [h, 0.5,- 0.5], [h, -0.5, 0.5]]
)
np_triangles = np.array([[0, 1, 2], [0, 1, 3]]).astype(np.int32)
plane_top.vertices = o3d.utility.Vector3dVector(np_vertices)
plane_top.triangles = o3d.utility.Vector3iVector(np_triangles)
plane_top.paint_uniform_color([1, 0, 0])

N = 6890
left_arm_list = []
right_arm_list = []
left_leg_list = []
right_leg_list = []
torso_list = []

for i in range(N):
    point = xyz[i]
    if (point[0] > 0.18 and point[1] > -0.2):
        left_arm_list.append(i)
    elif (point[0] < -0.18 and point[1] > 0) or (point[0] < -0.20 and point[1] > -0.3):
        right_arm_list.append(i)
    elif (point[0] > -0.015 and point[0] < 0.3 and point[1] < -0.1):
        left_leg_list.append(i)
    elif (point[0] < -0.015 and point[0] > -0.3 and point[1] < -0.1):
        right_leg_list.append(i)
    else:
        torso_list.append(i)

print(right_arm_list)
xyz_left_arm = xyz[left_arm_list]
xyz_right_arm = xyz[right_arm_list]
xyz_left_leg = xyz[left_leg_list]
xyz_right_leg = xyz[right_leg_list]
xyz_torso = xyz[torso_list]

# Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
left_arm_pcd = o3d.geometry.PointCloud()
left_arm_pcd.points = o3d.utility.Vector3dVector(xyz_left_arm)
left_arm_pcd.paint_uniform_color([0,0,1])
right_arm_pcd = o3d.geometry.PointCloud()
right_arm_pcd.points = o3d.utility.Vector3dVector(xyz_right_arm)
right_arm_pcd.paint_uniform_color([0,1,1])
left_leg_pcd = o3d.geometry.PointCloud()
left_leg_pcd.points = o3d.utility.Vector3dVector(xyz_left_leg)
left_leg_pcd.paint_uniform_color([1,0,1])
right_leg_pcd = o3d.geometry.PointCloud()
right_leg_pcd.points = o3d.utility.Vector3dVector(xyz_right_leg)
right_leg_pcd.paint_uniform_color([0,1,0])
torso_pcd = o3d.geometry.PointCloud()
torso_pcd.points = o3d.utility.Vector3dVector(xyz_torso)
torso_pcd.paint_uniform_color([1,0,0])
pcd.paint_uniform_color([1,0.5,0])

o3d.visualization.draw([left_arm_pcd, right_arm_pcd, left_leg_pcd, right_leg_pcd, torso_pcd])