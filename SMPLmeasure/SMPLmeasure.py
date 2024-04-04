import json

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from skspatial.objects import Plane, Points


class SMPLmeasure:
    """Measure SMPL"""

    def __init__(self):

        with open("./SMPLmeasure/SMPL_index_measure.json") as json_file:
            self.indexes = json.load(json_file)

        with open("./SMPLmeasure/SMPL_face_index_body_parts.json") as json_file:
            self.body_parts = json.load(json_file)

    def _compute_convex_hull(self, points):
        """
        Compute the convex hull from a 2D numpy array of points.

        Parameters:
            points (numpy.ndarray): Array of shape (n, 2) representing 2D points.

        Returns:
            numpy.ndarray: Array of shape (m, 2) representing the vertices of the convex hull.
        """
        # Ensure the input is a numpy array
        points = np.array(points)

        # Compute the convex hull
        hull = ConvexHull(points)

        # Get the vertices of the convex hull
        hull_vertices = points[hull.vertices]

        return hull_vertices

    def _get_polar_angle(self, point, center):
        """Calculate the polar angle of a point with respect to the center."""
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return np.arctan2(dy, dx)

    def _order_points(self, selected_points):
        """Order the points based on their polar angles with respect to the centroid."""
        # Find the centroid of the figure
        centroid = np.mean(selected_points, axis=0)
        
        # Calculate polar angles for all points
        polar_angles = [self._get_polar_angle(point, centroid) for point in selected_points]
        
        # Sort points based on polar angles
        ordered_indices = np.argsort(polar_angles)
        
        # Reorder the points
        ordered_points = selected_points[ordered_indices]
        
        return ordered_points


    def _compute_length(self, mesh, xyz, compute_convex_hull=False):
        points = Points(xyz)
        plane = Plane.best_fit(points)
        normal = np.array(plane.normal)

        isect, face_inds = trimesh.intersections.mesh_plane(
            mesh,
            plane_normal=normal,
            plane_origin=plane.project_point(xyz[0]),
            return_faces=True
        )

        curve = np.array(isect[:,0,0::2])
        ordered_curve = self._order_points(curve)

        if compute_convex_hull==True:
            ordered_curve = self._compute_convex_hull(ordered_curve)

        length = 0

        for i in range(1, ordered_curve.shape[0]):
            length += np.linalg.norm(ordered_curve[i] - ordered_curve[i - 1])

        length += np.linalg.norm(ordered_curve[-1] - ordered_curve[0])

        return length

    def measure_smpl(self, filename, height):

        mesh = trimesh.load(filename)
        xyz = mesh.vertices

        real_height = height
        upper_y = np.mean(xyz[self.indexes["head_tip"]][:, 1])
        lower_y = np.mean(xyz[self.indexes["feet_soles"]][:, 1])
        smpl_height = upper_y - lower_y

        left_thigh = self._compute_length(mesh.submesh([self.body_parts["left_leg"]], append=True), xyz[self.indexes["left_thigh"]]) * real_height / smpl_height
        right_thigh = self._compute_length(mesh.submesh([self.body_parts["right_leg"]], append=True), xyz[self.indexes["right_thigh"]]) * real_height / smpl_height
        left_calf = self._compute_length(mesh.submesh([self.body_parts["left_leg"]], append=True), xyz[self.indexes["left_calf"]]) * real_height / smpl_height
        right_calf = self._compute_length(mesh.submesh([self.body_parts["right_leg"]], append=True), xyz[self.indexes["right_calf"]]) * real_height / smpl_height
        chest = self._compute_length(mesh.submesh([self.body_parts["torso"]], append=True), xyz[self.indexes["chest"]]) * real_height / smpl_height
        waist = self._compute_length(mesh.submesh([self.body_parts["torso"]], append=True), xyz[self.indexes["waist"]]) * real_height / smpl_height
        hip = self._compute_length(mesh.submesh([self.body_parts["torso"]], append=True), xyz[self.indexes["hip"]]) * real_height / smpl_height
        left_arm = self._compute_length(mesh.submesh([self.body_parts["left_arm"]], append=True), xyz[self.indexes["left_arm"]]) * real_height / smpl_height
        right_arm = self._compute_length(mesh.submesh([self.body_parts["right_arm"]], append=True), xyz[self.indexes["right_arm"]]) * real_height / smpl_height

        results = {
            "left_thigh": left_thigh,
            "right_thigh": right_thigh,
            "left_calf": left_calf,
            "right_calf": right_calf,
            "chest": chest,
            "waist": waist,
            "hip": hip,
            "left_arm": left_arm,
            "right_arm": right_arm,
        }

        return results
