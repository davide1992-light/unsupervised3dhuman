import torch
from pytorch3d.structures import Meshes, Pointclouds

from SMPLfitter.src.utils import chamfer_distance_sep, point_mesh_face_distance_sep


# Guassian
def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x**2
    sigma_squared = sigma**2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


# angle prior
def angle_prior(pose):
    """
    Angle prior that penalizes unnatural bending of the knees and elbows
    """
    # We subtract 3 because pose does not include the global rotation of the model
    return torch.exp(pose[:, [55 - 3, 58 - 3, 12 - 3, 15 - 3]] * torch.tensor([1.0, -1.0, -1, -1.0], device=pose.device)) ** 2


# ----- use body fitting with index EM ----------
def body_fitting_loss_em(
    body_pose,
    preserve_pose,
    betas,
    preserve_betas,
    modelVerts,
    meshVerts,
    modelInd,
    meshInd,
    probInput,
    pose_prior,
    vertices,
    modelfaces,
    sigma=100,
    pose_prior_weight=4.78,
    shape_prior_weight=5.0,
    angle_prior_weight=15.2,
    betas_preserve_weight=10.0,
    pose_preserve_weight=10.0,
    chamfer_weight=2000.0,
    correspond_weight=800.0,
    point2mesh_weight=5000.0,
):
    """
    Loss function for body fitting
    """

    probInputM = torch.repeat_interleave(torch.reshape(probInput, (1, -1, 1)), 3, dim=2)
    correspond_loss = correspond_weight * (probInputM * gmof(modelVerts[:, modelInd] - meshVerts[:, meshInd], sigma)).sum()

    # Pose prior loss
    pose_prior_loss = (pose_prior_weight**2) * pose_prior(body_pose, betas)
    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight**2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight**2) * (betas**2).sum(dim=-1)

    betas_preserve_loss = (betas_preserve_weight**2) * ((betas - preserve_betas) ** 2).sum(dim=-1)
    pose_preserve_loss = (pose_preserve_weight**2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

    chamfer_loss = (chamfer_weight**2) * get_chamfer_loss(meshVerts, modelVerts)[0] + ((chamfer_weight / 4.0) ** 2) * get_chamfer_loss(meshVerts, modelVerts)[1]

    point2mesh_loss = (point2mesh_weight**2) * get_point2mesh_loss(meshVerts, vertices, 1, modelfaces)

    total_loss = correspond_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + betas_preserve_loss + pose_preserve_loss + chamfer_loss + point2mesh_loss

    return total_loss.sum()


def get_chamfer_loss(point_arr, output):
    chamfer_x, chamfer_y, _, _ = chamfer_distance_sep(point_arr, output, batch_reduction="sum")
    return chamfer_x, chamfer_y


def get_point2mesh_loss(point_arr, vertices, batchSize, modelface):
    points_list = []
    verts_list = []
    faces_list = []
    for idx in range(batchSize):
        points_list.append(point_arr[idx].squeeze())
        verts_list.append(vertices[idx].squeeze())
        faces_list.append(modelface)
    meshes = Meshes(verts_list, faces_list)
    pcls = Pointclouds(points_list)

    point2mesh_loss, _ = point_mesh_face_distance_sep(meshes, pcls)

    return point2mesh_loss
