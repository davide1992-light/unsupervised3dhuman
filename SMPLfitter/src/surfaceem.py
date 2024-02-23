import json

import numpy as np
import torch
from tqdm import tqdm

from SMPLfitter.src.customloss import body_fitting_loss_em
from SMPLfitter.src.prior import MaxMixturePrior


#  surface EM
class surface_EM_depth:
    """Implementation of SMPLify, use surface."""

    def __init__(
        self,
        smpl_model,
        learning_rate=1e-1,
        batch_size=1,
        num_iters=100,
        selected_index=np.arange(6890),
        device=torch.device("cuda:0"),
        mu=0.05,
    ):

        # Store options
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder="./SMPLfitter/smpl_models/", num_gaussians=8, dtype=torch.float32).to(device)
        # Load SMPL-X model
        self.smpl_model = smpl_model
        # self.modelfaces = torch.from_numpy(np.int32(smplxmodel.faces)).to(device)
        self.selected_index = selected_index

        # mu prob
        self.mu = mu

    @torch.no_grad()
    def prob_cal(self, modelVerts_in, meshVerts_in, sigma=0.05**2, mu=0.02):
        modelVerts = torch.squeeze(modelVerts_in)
        meshVerts = torch.squeeze(meshVerts_in)

        model_x, model_y, model_z = torch.split(modelVerts, [1, 1, 1], dim=1)
        mesh_x, mesh_y, mesh_z = torch.split(meshVerts, [1, 1, 1], dim=1)

        M = model_x.shape[0]
        N = mesh_x.shape[0]

        delta_x = torch.repeat_interleave(torch.transpose(mesh_x, 0, 1), M, dim=0) - torch.repeat_interleave(model_x, N, dim=1)
        delta_y = torch.repeat_interleave(torch.transpose(mesh_y, 0, 1), M, dim=0) - torch.repeat_interleave(model_y, N, dim=1)
        delta_z = torch.repeat_interleave(torch.transpose(mesh_z, 0, 1), M, dim=0) - torch.repeat_interleave(model_z, N, dim=1)

        deltaVerts = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z

        mu_c = ((2.0 * torch.asin(torch.tensor(1.0)) * sigma) ** (3.0 / 2.0) * mu * M) / ((1 - mu) * N)
        deltaExp = torch.exp(-deltaVerts / (2 * sigma))
        deltaExpN = torch.repeat_interleave(torch.reshape(torch.sum(deltaExp, dim=0), (1, N)), M, dim=0)
        probArray = deltaExp / (deltaExpN + mu_c)

        Ind = torch.where(probArray > 1e-6)  # 2e-7
        modelInd, meshInd = Ind[0], Ind[1]
        probInput = probArray[Ind]

        return probInput, modelInd, meshInd

    # ---- get the man function hrere
    def __call__(self, init_pose_front, init_pose_back, init_betas, init_scale, init_cam_trans_front, init_cam_trans_back, front_meshVerts, back_meshVerts):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose
            init_betas: SMPL betas
            init_cam_trans: Camera translation
            meshVerts: point3d from mesh
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        global_orient_front = init_pose_front[:, :3].detach().clone()
        body_pose_front = init_pose_front[:, 3:].detach().clone()
        global_orient_back = init_pose_back[:, :3].detach().clone()
        body_pose_back = init_pose_back[:, 3:].detach().clone()
        scale = init_scale.detach().clone()
        betas = init_betas.detach().clone()
        camera_translation_front = init_cam_trans_front.clone()
        camera_translation_back = init_cam_trans_back.clone()

        preserve_betas = init_betas.detach().clone()
        preserve_pose_front = init_pose_front[:, 3:].detach().clone()
        preserve_pose_back = init_pose_back[:, 3:].detach().clone()

        global_orient_front.requires_grad = True
        body_pose_front.requires_grad = True
        global_orient_back.requires_grad = True
        body_pose_back.requires_grad = True
        betas.requires_grad = True
        scale.requires_grad = True
        camera_translation_front.requires_grad = True
        camera_translation_back.requires_grad = True
        body_opt_params = [body_pose_front, global_orient_front, body_pose_back, global_orient_back, betas, scale, camera_translation_front, camera_translation_back]
        body_optimizer = torch.optim.LBFGS(body_opt_params, max_iter=20, lr=self.learning_rate, line_search_fn="strong_wolfe")

        store_epoch = {"loss": [], "betas": [], "global_orient": [], "body_pose": [], "scale": [], "camera_translation": [], "sigma": []}

        for i in tqdm(range(self.num_iters)):

            def closure():
                body_optimizer.zero_grad()
                
                vertices_front, faces = self.smpl_model(
                    beta=betas.squeeze(),
                    pose=torch.cat([global_orient_front, body_pose_front], dim=-1).squeeze(),
                    trans=camera_translation_front.squeeze(),
                )

                faces = torch.from_numpy(np.int32(faces)).to(self.device)

                vertices_back, _ = self.smpl_model(
                    beta=betas.squeeze(),
                    pose=torch.cat([global_orient_back, body_pose_back], dim=-1).squeeze(),
                    trans=camera_translation_back.squeeze(),
                )
                vertices_front = vertices_front.unsqueeze(0)
                vertices_back = vertices_back.unsqueeze(0)

                modelVerts_front = vertices_front[:, self.selected_index]
                scaled_modelVerts_front = torch.mul(modelVerts_front, scale)

                modelVerts_back = vertices_back[:, self.selected_index]
                scaled_modelVerts_back = torch.mul(modelVerts_back, scale)

                sigma = (0.1**2) * (self.num_iters - i + 1) / self.num_iters
                front_probInput, front_modelInd, front_meshInd = self.prob_cal(scaled_modelVerts_front, front_meshVerts, sigma=sigma, mu=self.mu)
                back_probInput, back_modelInd, back_meshInd = self.prob_cal(scaled_modelVerts_back, back_meshVerts, sigma=sigma, mu=self.mu)

                betas_preserve_weight = 3.0
                shape_prior_weight = 5.0

                front_loss = body_fitting_loss_em(
                    body_pose_front,
                    preserve_pose_front,
                    betas,
                    preserve_betas,
                    scaled_modelVerts_front,
                    front_meshVerts,
                    front_modelInd,
                    front_meshInd,
                    front_probInput,
                    self.pose_prior,
                    vertices_front,
                    faces,
                    pose_prior_weight=4.78 * 2.0,
                    shape_prior_weight=shape_prior_weight,
                    angle_prior_weight=15.2,
                    betas_preserve_weight=betas_preserve_weight,
                    pose_preserve_weight=1.0,
                    chamfer_weight=200.0,
                    correspond_weight=1500.0,
                    point2mesh_weight=0.0,
                )
                back_loss = body_fitting_loss_em(
                    body_pose_back,
                    preserve_pose_back,
                    betas,
                    preserve_betas,
                    scaled_modelVerts_back,
                    back_meshVerts,
                    back_modelInd,
                    back_meshInd,
                    back_probInput,
                    self.pose_prior,
                    vertices_back,
                    faces,
                    pose_prior_weight=4.78 * 2.0,
                    shape_prior_weight=shape_prior_weight,
                    angle_prior_weight=15.2,
                    betas_preserve_weight=betas_preserve_weight,
                    pose_preserve_weight=1.0,
                    chamfer_weight=200.0,
                    correspond_weight=1500.0,
                    point2mesh_weight=0.0,
                )

                loss = front_loss + back_loss
                loss.backward()
                return loss

            loss = body_optimizer.step(closure)

            store_epoch["loss"].append(loss.tolist()),
            store_epoch["betas"].append(betas.tolist()[0]),
            store_epoch["global_orient"].append(global_orient_front.tolist()[0]),
            store_epoch["body_pose"].append(body_pose_front.tolist()[0]),
            store_epoch["scale"].append(scale.tolist()),
            store_epoch["camera_translation"].append(camera_translation_front.tolist()[0])
            store_epoch["sigma"].append((0.1**2) * (self.num_iters - i + 1) / self.num_iters)

            with open("./store_epoch.json", "w") as f:
                json.dump(store_epoch, f)

        pose_front = torch.cat([global_orient_front, body_pose_front], dim=-1).detach()
        pose_back = torch.cat([global_orient_back, body_pose_back], dim=-1).detach()
        betas = betas.detach()
        scale = scale.detach()
        camera_translation_front = camera_translation_front.detach()
        camera_translation_back = camera_translation_back.detach()

        return pose_front, pose_back, betas, scale, camera_translation_front, camera_translation_back
