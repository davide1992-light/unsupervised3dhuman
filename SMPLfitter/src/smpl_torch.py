import pickle

import numpy as np
import torch
from torch.nn import Module


class SMPL(Module):
    def __init__(self, model_path, device=None):
        super(SMPL, self).__init__()

        with open(model_path, "rb") as f:
            params = pickle.load(f)

        self.J_regressor = torch.from_numpy(np.array(params["J_regressor"].todense())).type(torch.float32)
        self.weights = torch.from_numpy(params["weights"]).type(torch.float32)
        self.posedirs = torch.from_numpy(params["posedirs"]).type(torch.float32)
        self.v_template = torch.from_numpy(params["v_template"]).type(torch.float32)
        self.shapedirs = torch.from_numpy(params["shapedirs"]).type(torch.float32)
        self.kintree_table = params["kintree_table"]
        self.f = params["f"]

        self.device = device if device is not None else torch.device("cpu")
        for name in ["J_regressor", "weights", "posedirs", "v_template", "shapedirs"]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

    @staticmethod
    def _rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        # r = r.to(self.device)
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=torch.float32).to(r.device)
        m = torch.stack(
            (
                z_stick,
                -r_hat[:, 0, 2],
                r_hat[:, 0, 1],
                r_hat[:, 0, 2],
                z_stick,
                -r_hat[:, 0, 0],
                -r_hat[:, 0, 1],
                r_hat[:, 0, 0],
                z_stick,
            ),
            dim=1,
        )
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (
            torch.eye(3, dtype=torch.float32).unsqueeze(dim=0) + torch.zeros((theta_dim, 3, 3), dtype=torch.float32)
        ).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def _with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.

        Parameter:
        ---------
        x: Tensor to be appended.

        Return:
        ------
        Tensor after appending of shape [4,4]

        """
        ones = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).to(x.device)
        ret = torch.cat((x, ones), dim=0)
        return ret

    @staticmethod
    def _pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.

        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]

        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.

        """
        zeros43 = torch.zeros((x.shape[0], 4, 3), dtype=torch.float32).to(x.device)
        ret = torch.cat((zeros43, x), dim=2)
        return ret

    def forward(self, beta, pose, trans, simplify=False):
        """
        Construct a compute graph that takes in parameters and outputs a tensor as
        model vertices. Face indices are also returned as a numpy ndarray.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] tensor indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A tensor of shape [10] as coefficients of
        PCA components. Only 10 components were released by SMPL author.

        trans: Global translation tensor of shape [3].

        Return:
        ------
        A tensor for vertices, and a numpy ndarray as face indices.

        """
        id_to_col = {self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])}
        parent = {i: id_to_col[self.kintree_table[0, i]] for i in range(1, self.kintree_table.shape[1])}
        v_shaped = torch.tensordot(self.shapedirs, beta, dims=([2], [0])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        R_cube_big = self._rodrigues(pose.view(-1, 1, 3))

        if simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[1:]
            I_cube = (
                torch.eye(3, dtype=torch.float32).unsqueeze(dim=0)
                + torch.zeros((R_cube.shape[0], 3, 3), dtype=torch.float32)
            ).to(self.device)
            lrotmin = torch.reshape(R_cube - I_cube, (-1, 1)).squeeze()
            v_posed = v_shaped + torch.tensordot(self.posedirs, lrotmin, dims=([2], [0]))

        results = []
        results.append(self._with_zeros(torch.cat((R_cube_big[0], torch.reshape(J[0, :], (3, 1))), dim=1)))
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[parent[i]],
                    self._with_zeros(
                        torch.cat(
                            (
                                R_cube_big[i],
                                torch.reshape(J[i, :] - J[parent[i], :], (3, 1)),
                            ),
                            dim=1,
                        )
                    ),
                )
            )

        stacked = torch.stack(results, dim=0)
        results = stacked - self._pack(
            torch.matmul(
                stacked,
                torch.reshape(
                    torch.cat(
                        (J, torch.zeros((24, 1), dtype=torch.float32).to(self.device)),
                        dim=1,
                    ),
                    (24, 4, 1),
                ),
            )
        )
        T = torch.tensordot(self.weights, results, dims=([1], [0]))
        rest_shape_h = torch.cat(
            (
                v_posed,
                torch.ones((v_posed.shape[0], 1), dtype=torch.float32).to(self.device),
            ),
            dim=1,
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (-1, 4, 1)))
        v = torch.reshape(v, (-1, 4))[:, :3]
        vertices = v + torch.reshape(trans, (1, 3))
        return vertices, self.f
