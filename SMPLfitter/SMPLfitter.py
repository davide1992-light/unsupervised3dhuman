from __future__ import division, print_function

import joblib
import smplx
import torch
import trimesh

from SMPLfitter.src.surfaceem import surface_EM_depth
from SMPLfitter.src.utils import farthest_point_sample


class SMPLfitter:
    """Implementation of SMPL fitter."""

    def __init__(
        self,
        smpl_gender="male",
    ):

        # Set device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        print("Selected device:", self.device)

        # SMPL gender
        if smpl_gender == "male" or smpl_gender == "female":
            self.smpl_gender = smpl_gender
        else:
            print('Wrong gender parameter, "male" or "female" accepted.')

        # Initialize SMPL model
        self.smpl_params_path = "./SMPLfitter/smpl_models/"
        self.smplmodel = smplx.create(self.smpl_params_path, model_type="smpl", gender=self.smpl_gender, ext="pkl").to(self.device)

        # Downsample index
        SMPL_downsample_index_path = "./SMPLfitter/smpl_models/SMPL_downsample_index.pkl"
        self.selected_index = joblib.load(SMPL_downsample_index_path)["downsample_index"]

    def load_pc(self, input_file: str) -> torch.tensor:
        """Load pointcloud
        Input:
        Output:
        """

        mesh = trimesh.load(input_file)
        points = mesh.vertices
        points = torch.from_numpy(points).float()
        print(f"Loaded {input_file} point cloud with {points.shape[0]} points")

        return points

    def sample_pc(self, points: torch.tensor) -> torch.tensor:
        """Sample point cloud to reduce number of points
        Input:
        Output:
        """

        index = farthest_point_sample(points.unsqueeze(0), npoint=2048).squeeze()  # Return sampled indexes from farthest_point_sample
        sampled_points = points[index]  # Select sampled indexes

        print("Sampled point cloud with %s points" % sampled_points.shape[0])
        return sampled_points

    def center_pc(self, points: torch.tensor) -> (torch.tensor, torch.tensor):
        """Center point cloud
        Input:
        Output:
        """

        center_trans = torch.mean(points, dim=0, keepdim=True)
        points = torch.sub(points, center_trans)

        print("Point cloud centered")
        return points, center_trans

    def initialize_params(self, center_trans: torch.tensor) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """Initilize Parameters
        Input:
        Output:
        """

        init_pose = torch.zeros(1, 72).to(self.device)
        init_betas = torch.zeros(1, 10).to(self.device)
        init_scale = torch.tensor(1, dtype=torch.float32).to(self.device)
        init_cam_trans = torch.zeros(1, 3).to(self.device)
        center_trans = center_trans.to(self.device)

        print("Parameters initialized initialized")
        return init_pose, init_betas, init_scale, init_cam_trans, center_trans

    def smpl_fit(
        self,
        points: torch.tensor,
        init_pose: torch.tensor,
        init_betas: torch.tensor,
        init_scale: torch.tensor,
        init_cam_trans: torch.tensor,
    ) -> (torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor):
        """Fit SMPL model
        Input:
        Output:
        """

        points = points.unsqueeze(0).to(self.device)
        depthEM = surface_EM_depth(
            smplxmodel=self.smplmodel,
            batch_size=1,
            num_iters=100,
            selected_index=self.selected_index,
            device=self.device,
        )

        pred_pose, pred_betas, pred_scale, pred_cam_trans = depthEM(init_pose.detach(), init_betas.detach(), init_scale.detach(), init_cam_trans.detach(), points)

        print("SMPL parameters fitted")
        return pred_pose, pred_betas, pred_scale, pred_cam_trans

    def save_smpl_ply(
        self,
        pose: torch.tensor,
        betas: torch.tensor,
        scale: torch.tensor,
        cam_trans: torch.tensor,
        center_trans: torch.tensor,
        filename: str,
    ) -> None:
        """Save SMPL point cloud
        Input:
        Output:
        """

        output = self.smplmodel(
            betas=betas,
            global_orient=pose[:, :3],
            body_pose=pose[:, 3:],
            transl=cam_trans + center_trans,
            return_verts=True,
        )
        scaled_outputVerts = torch.mul(output.vertices, scale).detach().cpu().numpy().squeeze()
        mesh = trimesh.Trimesh(vertices=scaled_outputVerts, faces=self.smplmodel.faces, process=False)
        mesh.export(filename)

        print(f"Predicted SMPL saved to {filename}")
        return None
