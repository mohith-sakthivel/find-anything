import os
import random
import numpy as np
import open3d as o3d
from tqdm import tqdm
import time

import torch
from torch.utils.data import Dataset

class FindAnythingDataset(Dataset):
    """ Download data from 
        https://lmb.informatik.uni-freiburg.de/resources/datasets/ORION/modelnet40_manually_aligned.tar

        Args:
            root_dir (str): path to the root directory of the dataset, contianing classes at the top level
            split (str): "train" or "test" split of the dataset
            debug_mode (bool): if True, returns a list of open3d.geometry.PointCloud objects that were
                used for the scene that is returned (default is False)

        Returns Dictionary with keys:
            "query": Point cloud scene of shape (N, 3) in torch.float32, including a plane and a random
                number of objects from random classes
            "support_mask": Point cloud object to search of shape (M, 3) in torch.float32, is one of the 
                existing classes in the "query" scene
            "label": label mask for the support object within the query scene of shape (N, 1) in torch.uint8
                label mask is 1 if the point is part of the support object, 0 otherwise
            "instance_label": instance-level label mask for the support object within the query scene of
                shape (N, 1) in torch.uint8, beginning with 0 for the first instance, 1, and so on. All
                non-support points are labeled -1
            "debug_objects_list" (optional): list of open3d.geometry.PointCloud objects that were used for
                the scene that is returned if debug_mode is set to True (default is False), including the
                plane and all objects in the scene
    """
    def __init__(self, root_dir="data/ModelNet", split="train", debug_mode=False):
        # Set variables
        self.root_dir = root_dir
        self.split = split
        self.debug_mode = debug_mode

        # Get all types of objects
        self.obj_classes = [obj_class for obj_class in os.listdir(root_dir)]

        # Hyperparameters
        self.min_scene_instances = 6
        self.max_scene_instances = 10
        self.min_target_instances = 1
        self.max_target_instances = 3
        self.point_density_by_area = 10
        self.points_on_plane = 300
        self.object_size_range = [2, 3]
        self.plane_side_dim = 15

        # Create the ground plane to place objects on
        mesh_plane = o3d.geometry.TriangleMesh.create_box(width=self.plane_side_dim,
                                                          height=self.plane_side_dim,
                                                          depth=0.01)
        # Get the bounding box dimensions and center
        bbox = mesh_plane.get_axis_aligned_bounding_box()
        bbox_center = bbox.get_center()
        bbox_dims = bbox.get_max_bound() - bbox.get_min_bound()

        # Move the object so that the top of the bounding box is at z=0
        mesh_plane.translate([-bbox_center[0], -bbox_center[1], -bbox_dims[2]])

        # Turn it into a point cloud
        mesh_plane.compute_vertex_normals()
        self.mesh_pc = mesh_plane.sample_points_poisson_disk(number_of_points=self.points_on_plane,
                                                             init_factor=5)

        # Get list of all objects separated by train and test per class
        self.train_obj_dict = dict()
        self.test_obj_dict = dict()
        for obj_class in self.obj_classes:
            self.train_obj_dict[obj_class] = [obj for obj in os.listdir(
                os.path.join(root_dir, obj_class, "train")) if obj.endswith(".off")]
            self.test_obj_dict[obj_class] = [obj for obj in os.listdir(
                os.path.join(root_dir, obj_class, "test")) if obj.endswith(".off")]

    def __getitem__(self, idx):
        """
        Generate a scene from the dataset on top of a ground plane

        Notes
            A maximum of one object type per class is placed on the scene
            There can be multiple objects of the same type from one class
            Not all classes are guaranteed to be present in the scene
        """
        # Create a list of mesh objects to place in the scene
        objects_list = [self.mesh_pc]
        labels_list = [0] * self.points_on_plane
        instance_labels_list = [-1] * self.points_on_plane

        # Randomly select the number of classes to include in the scene
        num_scene_instances = random.randint(self.min_scene_instances, self.max_scene_instances)

        # Randomly determine one class to be the target class
        target_class = random.choice(self.obj_classes)

        # Remove the target class from list for sampling negative instances
        obj_classes_less_target = self.obj_classes.copy()  # Create a copy of the original list
        obj_classes_less_target.remove(target_class)  # Remove the item from the new list

        # Randomly sample colors for each class
        colors = [np.array([[0], [0], [0]])]

        # Randomly select the number of target instances to include in the scene
        num_target_instances = random.randint(self.min_target_instances, self.max_target_instances)

        # First sample target instance
        if self.split == "train":
            obj = random.choice(self.train_obj_dict[target_class])
        else:
            obj = random.choice(self.test_obj_dict[target_class])

        # Load the object
        mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, target_class, self.split, obj))
        
        # Randomly scale the object
        scale = random.uniform(self.object_size_range[0], self.object_size_range[1])

        # Scale the object to be the randomized scaling factor
        mesh.scale(scale / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())

        # Get surface area of the mesh
        mesh_sa = mesh.get_surface_area()
        points_to_sample = int(mesh_sa * self.point_density_by_area)

        # Add class binary label to list number of times equal to num_objects
        labels_list.extend([1] * num_target_instances * points_to_sample)

        # Get the bounding box dimensions and center
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_center = bbox.get_center()
        bbox_dims = bbox.get_max_bound() - bbox.get_min_bound()

        # Move the object so that the bounding box center is above the origin and the 
        # bottom of the bounding box is at z=0
        mesh.translate(-bbox_center)
        mesh.translate([0, 0, bbox_dims[2] / 2])

        # Target color is yellow
        color = np.array([[1], [1], [0]])

        # Loop through each object and add it to the scene
        for idx in range(num_target_instances):
            # Sample points from the mesh
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_poisson_disk(number_of_points=points_to_sample, 
                                                    init_factor=5)

            # Define a random position and orientation for the object - no randomness in position currently
            z_rot = random.uniform(0, 360)
            rotation = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, z_rot])

            # Create transformation matrix
            transform = np.zeros((4, 4))
            transform[:3, :3] = rotation
            transform[3, 3] = 1

            # Apply transformation to the point cloud
            pcd = pcd.transform(transform)

            # Add the object to the scene
            objects_list.append(pcd)

            # Add object instance label to list number of times equal to num_objects
            instance_labels_list.extend([idx] * points_to_sample)

            # Add the object to the target list
            if idx == num_target_instances - 1:
                support_pcd = mesh.sample_points_poisson_disk(number_of_points=points_to_sample, 
                                                                init_factor=5)

            colors.append(color)
        
        # 2. Sample negative instances
        for idx in range(num_scene_instances - num_target_instances):
            # Sample negative instance class, then sample object from that class
            neg_class = random.choice(obj_classes_less_target)
            if self.split == "train":
                obj = random.choice(self.train_obj_dict[neg_class])
            else:
                obj = random.choice(self.test_obj_dict[neg_class])

            # Load the object
            mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, neg_class, self.split, obj))
            
            # Randomly scale the object
            scale = random.uniform(self.object_size_range[0], self.object_size_range[1])

            # Scale the object to be the randomized scaling factor
            mesh.scale(scale / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())

            # Get surface area of the mesh
            mesh_sa = mesh.get_surface_area()
            points_to_sample = int(mesh_sa * self.point_density_by_area)

            # Add class binary label to list number of times equal to num_objects
            labels_list.extend([0] * points_to_sample)

            # Get the bounding box dimensions and center
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_center = bbox.get_center()
            bbox_dims = bbox.get_max_bound() - bbox.get_min_bound()

            # Move the object so that the bounding box center is above the origin and the 
            # bottom of the bounding box is at z=0
            mesh.translate(-bbox_center)
            mesh.translate([0, 0, bbox_dims[2] / 2])

            # Sample points from the mesh
            mesh.compute_vertex_normals()
            pcd = mesh.sample_points_poisson_disk(number_of_points=points_to_sample, 
                                                    init_factor=5)

            # Define a random position and orientation for the object - no randomness in position currently
            z_rot = random.uniform(0, 360)
            rotation = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, z_rot])

            # Create transformation matrix
            transform = np.zeros((4, 4))
            transform[:3, :3] = rotation
            transform[3, 3] = 1

            # Apply transformation to the point cloud
            pcd = pcd.transform(transform)

            # Add the object to the scene
            objects_list.append(pcd)

            # Add object instance label to list number of times equal to num_objects
            instance_labels_list.extend([-1] * points_to_sample)

            # Add color
            colors.append(np.random.rand(3,1))

        # Move objects into a grid-like pattern
        num_rows = int(np.sqrt(len(objects_list) - 1))
        num_cols = int((np.ceil(len(objects_list) - 1) / num_rows))
        spacing = self.plane_side_dim / (num_cols + 1)

        # Move the objects into a grid-like pattern with randomly sampled ordering for translation
        ordering = np.arange(0, len(objects_list) + 1)
        np.random.shuffle(ordering[1:])
        for i, obj in enumerate(objects_list):
            if i == 0:
                continue
            order_Idx = ordering[i]
            row = (order_Idx - 1) % num_rows
            col = (order_Idx - 1) // num_rows
            x = (col - num_cols / 2) * spacing
            y = (row - num_rows / 2) * spacing
            obj.translate([x, y, 0])

        # Create query point cloud
        object_tensors_list = []
        for object in objects_list:
            object_tensor = torch.from_numpy(np.concatenate((np.asarray(object.points),
                                                      np.asarray(object.normals)),
                                                      axis=1)).type(torch.float32)
            object_tensors_list.append(object_tensor)
        query_pc = torch.cat(object_tensors_list, dim=0)

        # Create labels and instance labels
        labels = torch.from_numpy(np.asarray(labels_list)).type(torch.uint8)
        instance_labels = torch.from_numpy(np.asarray(instance_labels_list)).type(torch.uint8)

        # Create support point cloud
        support_pc = torch.from_numpy(np.concatenate((np.asarray(support_pcd.points),
                                                      np.asarray(support_pcd.normals)),
                                                      axis=1)).type(torch.float32)
        
        # Combine all outputs and then shuffle all the rows
        combined = torch.cat((query_pc, labels.unsqueeze(1), instance_labels.unsqueeze(1)), dim=1)
        combined = combined[torch.randperm(combined.shape[0])]
        query_pc = combined[:, :6]
        labels = combined[:, 6]
        instance_labels = combined[:, 7]

        # Shuffle support_pc by row separately as it has a different size
        support_pc = support_pc[torch.randperm(support_pc.shape[0])]

        # Return query point cloud, support point cloud, labels, and instance labels as dictionary
        data = {"query": query_pc,
                "support_mask": support_pc,
                "label": labels,
                "instance_label": instance_labels}
        
        if self.debug_mode is True:
            data["debug_objects_list"] = objects_list
            data["colors"] = colors

        return data
    
    def __len__(self):
        return 100

if __name__ == "__main__":
    # Create a dataset object
    dataset = FindAnythingDataset(debug_mode=True)

    # Visualize one point cloud from the dataset
    data = dataset[0]

    # Create an open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([1, 1, 1])

    total_num_points = 0

    # Render the scene
    for obj, color in zip(data['debug_objects_list'], data["colors"]):
        obj.paint_uniform_color(color)
        vis.add_geometry(obj)
        total_num_points += np.asarray(obj.points).shape[0] 
    print("Total number of points (including plane):", total_num_points)
    print("Number of objects (excluding plane):", len(data['debug_objects_list']) - 1)
    vis.run()
    vis.destroy_window()