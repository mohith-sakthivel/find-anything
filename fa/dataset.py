import os
import random
import matplotlib
import numpy as np

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.io import IO
import pytorch3d.transforms as t3d

import torch
from torch.utils.data import Dataset


class FindAnythingDataset(Dataset):
    """Download data from
    https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar

    Args:
        root_dir (str): path to the root directory of the dataset, contianing classes at the top level
        split (str): "train" or "test" split of the dataset
        debug_mode (bool): if True, returns a list of open3d.geometry.PointCloud objects that were
            used for the scene that is returned (default is False)

    Returns Dictionary with keys:
        "scene": Point cloud scene of shape (N, 3) in torch.float32, including a plane and a random
            number of objects from random classes
        "template_mask": Point cloud object to search of shape (M, 3) in torch.float32, is one of the
            existing classes in the scene
        "label": label mask for the template object within the scene of shape (N, 1) in torch.uint8
            label mask is 1 if the point is part of the template object, 0 otherwise
        "instance_label": instance-level label mask for the template object within the scene of
            shape (N, 1) in torch.uint8, beginning with 0 for the first instance, 1, and so on. All
            non-template points are labeled -1
        "debug_objects_list" (optional): list of open3d.geometry.PointCloud objects that were used for
            the scene that is returned if debug_mode is set to True (default is False), including the
            plane and all objects in the scene
    """

    SAMPLING_UPSCALE_FACTOR = 1.25
    PLANE_DOWNSCALE_FACTOR = 2
    DATA_MEAN = [0, 0, 1.5, 0, 0, 0]
    DATA_STD = [6, 6, 1.5, 1, 1, 1]

    def __init__(
        self,
        root_dir: str = "data/ModelNet",
        split: str = "train",
        dataset_size: int = 1e4,
        num_scene_points: int = 2048,
        num_template_points: int = 1024,
        debug_mode: int = False,
        use_normals_for_scene: bool = True,
        use_normals_for_template: bool = True,
        rotate_template_pc: bool = False,
    ):
        # Set variables
        self.root_dir = root_dir
        self.split = split
        self.dataset_size = int(dataset_size)
        self.debug_mode = debug_mode
        self.num_scene_points = num_scene_points
        self.num_template_points = num_template_points
        self.use_normals_for_scene = use_normals_for_scene
        self.use_normals_for_template = use_normals_for_template
        self.scene_pc_dim = 6 if self.use_normals_for_scene else 3
        self.template_pc_dim = 6 if self.use_normals_for_template else 3
        self.rotate_template_pc = rotate_template_pc

        # Get all types of objects
        self.obj_classes = [obj_class for obj_class in os.listdir(root_dir)]

        # Hyperparameters
        self.min_scene_instances = 6
        self.max_scene_instances = 10
        self.min_target_instances = 1
        self.max_target_instances = 3
        self.object_size_range = [2, 3]
        self.plane_side_dim = 13
        self.degen_volume_sa_ratio = 200
        self.degen_max_min_dim_ratio = 15

        self.cmap = matplotlib.colormaps["jet"]
        self.data_mean = torch.tensor(self.DATA_MEAN, dtype=torch.float32)
        self.data_std = torch.tensor(self.DATA_STD, dtype=torch.float32)

        # Get list of all objects separated by train and test per class
        self.train_obj_dict = dict()
        self.test_obj_dict = dict()
        for obj_class in self.obj_classes:
            self.train_obj_dict[obj_class] = [
                obj for obj in os.listdir(os.path.join(root_dir, obj_class, "train")) if obj.endswith(".off")
            ]
            self.test_obj_dict[obj_class] = [
                obj for obj in os.listdir(os.path.join(root_dir, obj_class, "test")) if obj.endswith(".off")
            ]

    def create_plane(self, side_dim):
        # Generate vertices of the plane centered at (0,0,0)
        half_side = side_dim / 2.0
        vertices = torch.tensor(
            [[-half_side, -half_side, 0], [half_side, -half_side, 0], [half_side, half_side, 0], [-half_side, half_side, 0]]
        )
        # Generate triangular faces of the plane
        triangles = torch.tensor([[0, 1, 2], [0, 2, 3]])
        # Create an Open3D TriangleMesh object
        plane = Meshes(verts=[vertices], faces=[triangles])
        plane._compute_vertex_normals()
        return plane

    def random_positions(self, side_dim, obj_point_clouds, obj_counts, discretization_factor=0.1):
        # Create a 0.1 discretized boolean 2D array with the same side as the side_dim initialized to 1's
        grid = np.ones(
            (int(side_dim / discretization_factor), int(side_dim / discretization_factor)),
            dtype=bool
        )

        # Sample index positions from 1's and set the surrounding area (max possible size * 1.5) to 0's
        positions = []
        for i in range(len(obj_point_clouds) - 1):
            for j in range(obj_counts[i]):
                # Find indices where the array has a value of 1
                indices = np.argwhere(grid == True)

                # If no indices with value 1 are found, return None
                assert len(indices) > 0, "Error: No indices with value 1 found"

                # Randomly select an index from the list of indices
                sampled_index = indices[np.random.randint(0, len(indices))]

                # Set the surrounding area to 0's, 0.75 x max_obj_side_dim in each direction
                bbox = obj_point_clouds[i].get_bounding_boxes().squeeze()
                max_obj_side_dim = max(bbox[:, 1] - bbox[:, 0])
                offset = int(np.ceil(max_obj_side_dim * 0.75 / discretization_factor))
                min_dim0 = max(0, sampled_index[0] - offset)
                max_min0 = min(grid.shape[0], sampled_index[0] + offset)
                min_dim1 = max(0, sampled_index[1] - offset)
                max_dim1 = min(grid.shape[1], sampled_index[1] + offset)
                grid[min_dim0:max_min0, min_dim1:max_dim1] = False

                # Calculate actual 2D position
                position = np.array(
                    [sampled_index[0] * 0.1 - side_dim / 2.0, sampled_index[1] * 0.1 - side_dim / 2.0, 0],
                    dtype=np.float32
                )
                positions.append(position)
        return positions

    def __getitem__(self, idx):
        """
        Generate a scene from the dataset on top of a ground plane

        Notes
            The target class and instance is randomly sampled
            All negative instances are from the non-target classes, and are randomly sampled
            Each scene will have:
                # target instances: num_target_instances
                # negative instances: num_scene_instances - num_target_instances
                + 1 plane
        """
        # Create a list of mesh objects to place in the scene
        obj_classes = []
        obj_class_ids = []
        obj_counts = []
        obj_meshes = []
        obj_surface_areas = []

        # Randomly determine one class to be the target class
        target_class_id = np.random.randint(low=0, high=len(self.obj_classes))

        # Randomly select the number of classes to include in the scene
        num_scene_instances = np.random.randint(low=self.min_scene_instances, high=self.max_scene_instances + 1)

        # Randomly select the number of target instances to include in the scene
        num_target_instances = random.randint(self.min_target_instances, min(self.max_target_instances, num_scene_instances))

        obj_class_ids.append(target_class_id)
        obj_classes.append(self.obj_classes[target_class_id])
        obj_counts.append(num_target_instances)

        # Remove the target class from list for sampling negative instances
        available_class_ids = [i for i in range(len(self.obj_classes)) if i != target_class_id]
        non_target_class_ids = np.random.choice(
            available_class_ids, size=(num_scene_instances - num_target_instances), replace=True
        )
        non_target_class_counts = {}
        for c in non_target_class_ids:
            non_target_class_counts[c] = non_target_class_counts.get(c, 0) + 1

        # Add to object counts
        for k, v in non_target_class_counts.items():
            obj_class_ids.append(k)
            obj_counts.append(v)
            obj_classes.append(self.obj_classes[k])

        # Add a color for each class
        obj_colors = [np.array(self.cmap(i))[:3] for i in np.linspace(0, 1, num=len(obj_classes) + 1)]
        for i in range(len(obj_classes)):
            while True:
                if self.split == "train":
                    obj = np.random.choice(self.train_obj_dict[obj_classes[i]])
                else:
                    obj = np.random.choice(self.test_obj_dict[obj_classes[i]])

                # Load the object
                file_path = os.path.join(self.root_dir, obj_classes[i], self.split, obj)
                mesh = IO().load_mesh(file_path)

                # Compute the volume / SA ratio to find degenerate cases
                bbox = mesh.get_bounding_boxes().squeeze()
                volume = (bbox[0, 1] - bbox[0, 0]) * (bbox[1, 1] - bbox[1, 0]) * (bbox[2, 1] - bbox[2, 0])
                surface_area = torch.sum(mesh.faces_areas_packed())
                ratio = volume / surface_area
                dim_ratio = max(bbox[:, 1] - bbox[:, 0]) / min(bbox[:, 1] - bbox[:, 0])

                # print("Volume / SA Ratio: ", obj_classes[i], ratio)
                # print("Dimension Ratio: ", obj_classes[i], dim_ratio)

                if ratio < self.degen_volume_sa_ratio and dim_ratio < self.degen_max_min_dim_ratio:
                    break

            # Randomly scale the object to a volume
            scale = random.uniform(self.object_size_range[0], self.object_size_range[1])

            # Scale the object to be the randomized scaling factor
            mesh_bbox = mesh.get_bounding_boxes().squeeze()
            # print("Mesh Bounding Box: ", mesh_bbox)
            mesh.scale_verts_((scale / torch.max(mesh_bbox[:, 1] - mesh_bbox[:, 0])).item())
            obj_surface_areas.append(torch.sum(mesh.faces_areas_packed()))
            # print("Scale ", scale / torch.max(mesh_bbox[:,1] - mesh_bbox[:,0]))
            # print("Mesh Bounds (Post): ", mesh.get_bounding_boxes().squeeze())

            # Get the bounding box dimensions and center
            bbox = mesh.get_bounding_boxes().squeeze()
            bbox_center = (bbox[:, 0] + bbox[:, 1]) / 2.0
            bbox_dims = bbox[:, 1] - bbox[:, 0]

            # Move the object so that the bounding box center is above the origin and the
            # bottom of the bounding box is at z=0
            mesh.offset_verts_(-bbox_center)
            mesh.offset_verts_(torch.tensor([0, 0, bbox_dims[2] / 2]))

            mesh._compute_vertex_normals()
            obj_meshes.append(mesh)

        # Create a ground plane
        plane = self.create_plane(self.plane_side_dim)
        obj_classes.append("plane")
        obj_class_ids.append(len(self.obj_classes))
        obj_counts.append(1)
        obj_meshes.append(plane)
        obj_surface_areas.append(torch.sum(plane.faces_areas_packed()) / self.PLANE_DOWNSCALE_FACTOR)

        obj_surface_areas = np.array(obj_surface_areas)
        obj_counts = np.array(obj_counts)

        total_area = np.sum(obj_surface_areas * obj_counts)
        obj_num_pts = ((obj_surface_areas / total_area) * self.num_scene_points).astype(int)
        # Add residual points from integer division to last object
        obj_num_pts[-1] += self.num_scene_points - np.sum(obj_num_pts * obj_counts)

        obj_point_clouds = []
        for i in range(len(obj_meshes)):
            if obj_counts[i] == 1:
                num_samples = obj_num_pts[i]
            else:
                num_samples = int(self.SAMPLING_UPSCALE_FACTOR * obj_num_pts[i])

            pc = sample_points_from_meshes(obj_meshes[i], num_samples, return_normals=True)

            # Store the points (idx=0) and normals (idx=1)
            point_cloud = Pointclouds(points=pc[0], normals=pc[1])
            obj_point_clouds.append(point_cloud)

        scene_points = []
        scene_normals = []
        scene_class_labels = []
        scene_instance_labels = []
        scene_pt_colors = []
        # scene_meshes = []    # For visualization

        for i in range(len(obj_classes)):
            for j in range(obj_counts[i]):
                if i != len(obj_classes) - 1:
                    # Define a random orientation about the z-axis
                    z_rot = torch.tensor(random.uniform(0, 2 * np.pi))
                    rotation = t3d.euler_angles_to_matrix(torch.tensor([0.0, 0.0, z_rot]), "XYZ")
                    rotate = t3d.Rotate(rotation)

                    # Rotate the point cloud and normals
                    transformed_points = rotate.transform_points(obj_point_clouds[i].points_packed())
                    transformed_normals = rotate.transform_normals(obj_point_clouds[i].normals_packed())

                    transformed_pc = Pointclouds(points=transformed_points.unsqueeze(0), normals=transformed_normals.unsqueeze(0))

                    # # For visualization
                    # transformed_vertices = rotate.transform_points(obj_meshes[i].verts_packed())
                    # transformed_mesh = Meshes(transformed_vertices[None], obj_meshes[i].faces_packed()[None])

                else:
                    transformed_pc = obj_point_clouds[i]
                    # # For visualization
                    # transformed_mesh = Meshes(obj_meshes[i].verts_packed()[None], obj_meshes[i].faces_packed()[None])

                
                # scene_meshes.append(transformed_mesh)     # For visualization

                points = np.asarray(transformed_pc.points_packed(), dtype=np.float32)
                normals = np.asarray(transformed_pc.normals_packed(), dtype=np.float32)

                # If plane, set all normals to pointing straight up
                if i == len(obj_classes) - 1:
                    normals = np.zeros_like(normals)
                    normals[:, 2] = 1

                if obj_counts[i] > 1:
                    indices = np.random.choice(int(self.SAMPLING_UPSCALE_FACTOR * obj_num_pts[i]), obj_num_pts[i], replace=False)
                    points = points[indices]
                    normals = normals[indices]

                scene_points.append(points)
                scene_normals.append(normals)
                if i == 0:
                    scene_class_labels.append(np.ones(obj_num_pts[i], dtype=np.float32))
                else:
                    scene_class_labels.append(np.zeros(obj_num_pts[i], dtype=np.float32))
                scene_instance_labels.append(j * np.ones(obj_num_pts[i], dtype=np.float32))
                scene_pt_colors.append(obj_colors[i].reshape(1, 3).repeat(obj_num_pts[i], axis=0))

        # Randomly sample positions for all objects
        half_side_with_offset = self.plane_side_dim - self.object_size_range[1]
        random_positions = self.random_positions(half_side_with_offset, obj_point_clouds, obj_counts)
        for i in range(len(scene_points) - 1):
            scene_points[i] += random_positions[i]
            # For visualization
            # scene_meshes[i] = Meshes(scene_meshes[i].verts_packed()[None] + random_positions[i], scene_meshes[i].faces_packed()[None])
        
        # Create sample data
        scene_points = np.concatenate(scene_points, axis=0)
        scene_normals = np.concatenate(scene_normals, axis=0)
        if self.use_normals_for_scene:
            scene_pc = torch.from_numpy(np.concatenate([scene_points, scene_normals], axis=-1))
        else:
            scene_pc = torch.from_numpy(scene_points)
        class_labels = torch.from_numpy(np.concatenate(scene_class_labels, axis=0))
        instance_labels = torch.from_numpy(np.concatenate(scene_instance_labels, axis=0))

        # Create template point cloud
        sampled_pc = sample_points_from_meshes(obj_meshes[0], self.num_template_points, return_normals=True)
        if self.rotate_template_pc:
            # Define a random orientation
            xyz_rot = torch.tensor(np.random.uniform(0, 2 * np.pi, 3))
            rotation = t3d.euler_angles_to_matrix(xyz_rot, "XYZ")
            rotate = t3d.Rotate(rotation)
            # Rotate the point cloud and normals
            transformed_points = rotate.transform_points(sampled_pc[0])
            transformed_normals = rotate.transform_normals(sampled_pc[1])
            sampled_pc = (transformed_points, transformed_normals)

        if self.use_normals_for_template:
            template_pc = np.concatenate([np.asarray(sampled_pc[0].squeeze()), np.asarray(sampled_pc[1].squeeze())], axis=-1)
        else:
            template_pc = np.asarray(sampled_pc[0].squeeze())
        template_pc = torch.from_numpy(template_pc).to(torch.float32)

        # Normalize Data
        if self.use_normals_for_scene:
            scene_pc = (scene_pc - self.data_mean) / self.data_std
        else:
            scene_pc = (scene_pc - self.data_mean[:3]) / self.data_std[:3]

        if self.use_normals_for_template:
            template_pc = (template_pc - self.data_mean) / self.data_std
        else:
            template_pc = (template_pc - self.data_mean[:3]) / self.data_std[:3]

        # Return scene point cloud, template point cloud, labels, and instance labels as dictionary
        data = {
            "scene": scene_pc[: self.num_scene_points],
            "class_labels": class_labels[: self.num_scene_points],
            "instance_label": instance_labels[: self.num_scene_points],
            "template": template_pc,
        }

        if self.debug_mode is True:
            data["colors"] = np.concatenate(scene_pt_colors, axis=0)[: self.num_scene_points]
            data["obj_classes"] = obj_classes
            data["obj_counts"] = obj_counts

        return data

    def __len__(self) -> int:
        return self.dataset_size


if __name__ == "__main__":
    # Create a dataset object
    dataset = FindAnythingDataset(debug_mode=True)

    # Visualize one point cloud from the dataset
    data = dataset[0]
    print("Total number of points (including plane):", len(data["scene"]))
    for obj, obj_count in zip(data["obj_classes"], data["obj_counts"]):
        print(f"class: {obj:<15} \t count: {obj_count:02d}")

    # ### Visualize with matplotlib ###
    # import matplotlib.pyplot as plt

    # # Visualize the point cloud with matplotlib without Axes3D
    # point_cloud = data['scene']
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    # plt.show()

    # ### Test time ###
    # import time
    # from torch.utils.data import DataLoader

    # dataset = FindAnythingDataset(split="train")
    # dataloader = DataLoader(
    #     dataset=dataset,
    #     batch_size=8,
    #     num_workers=8,
    # )

    # start = time.time()
    # for data in dataloader:
    #     print(time.time() - start)
    #     start = time.time()
