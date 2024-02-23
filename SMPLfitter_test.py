import trimesh
import torch
from SMPLfitter import SMPLfitter

n = 2
front_input_file = f"./examples/ex{n}front_rot.ply"
back_input_file = f"./examples/ex{n}back_rot.ply"

fitter = SMPLfitter.SMPLfitter(smpl_gender="male")

front_points = fitter.load_pc(front_input_file)
front_sampled_points = fitter.sample_pc(front_points)
front_centered_points, front_center_trans = fitter.center_pc(front_sampled_points)

back_points = fitter.load_pc(back_input_file)
back_sampled_points = fitter.sample_pc(back_points)
back_centered_points, back_center_trans = fitter.center_pc(back_sampled_points)

init_pose_front, init_pose_back, init_betas, init_scale, init_cam_trans_front, init_cam_trans_back = fitter.initialize_params()

pred_pose_front, pred_pose_back, pred_betas, pred_scale, pred_cam_trans_front, pred_cam_trans_back = fitter.smpl_fit(front_centered_points, back_centered_points, init_pose_front, init_pose_back, init_betas, init_scale, init_cam_trans_front, init_cam_trans_back)

fitter.save_smpl_ply(pred_pose_front, pred_betas, pred_scale, pred_cam_trans_front, f"./results/ex{n}front_pred.ply")
fitter.save_smpl_ply(pred_pose_back, pred_betas, pred_scale, pred_cam_trans_back, f"./results/ex{n}back_pred.ply")