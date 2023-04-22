import os
import random
import matplotlib
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

    SAMPLING_UPSCALE_FACTOR = 1.25

    def __init__(
            self,
            root_dir: str = "data/ModelNet",
            split: str = "train",
            dataset_size: int = 1e4,
            num_query_points: int = 2048,
            num_support_points: int = 1024,
            debug_mode: int = False
        ):
        # Set variables
        self.root_dir = root_dir
        self.split = split
        self.dataset_size = int(dataset_size)
        self.debug_mode = debug_mode
        self.num_query_points = num_query_points
        self.num_support_points = num_support_points

        # Get all types of objects
        self.obj_classes = [obj_class for obj_class in os.listdir(root_dir)]

        # Hyperparameters
        self.min_scene_instances = 6
        self.max_scene_instances = 10
        self.min_target_instances = 1
        self.max_target_instances = 3
        self.points_on_plane = 300
        self.object_size_range = [2, 3]
        self.plane_side_dim = 15

        self.cmap = matplotlib.colormaps['jet']

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
        non_target_class_ids = np.random.choice(available_class_ids, size=(num_scene_instances - num_target_instances), replace=True)
        non_target_class_counts = {}
        for c in non_target_class_ids:
            non_target_class_counts[c] = non_target_class_counts.get(c, 0) + 1
        
        # Add to object counts
        for k, v in non_target_class_counts.items():
            obj_class_ids.append(k)
            obj_counts.append(v)
            obj_classes.append(self.obj_classes[k])

        # Add a color for each class
        obj_colors = [np.array(self.cmap(i))[:3] for i in np.linspace(0, 1, num=len(obj_classes))]

        for i in range(len(obj_classes)):
            while True:
                if self.split == "train":
                    obj = np.random.choice(self.train_obj_dict[obj_classes[i]])
                else:
                    obj = np.random.choice(self.test_obj_dict[obj_classes[i]])

                # Load the object
                mesh = o3d.io.read_triangle_mesh(os.path.join(self.root_dir, obj_classes[i], self.split, obj))
                
                # Calculate volume-to-surface area ratio
                volume = mesh.get_axis_aligned_bounding_box().volume()
                surface_area = mesh.get_surface_area()
                ratio = volume / surface_area
                
                if ratio < 3:
                    break

            # Randomly scale the object to a volume
            scale = random.uniform(self.object_size_range[0], self.object_size_range[1])

            # Scale the object to be the randomized scaling factor
            mesh.scale(scale / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
            obj_surface_areas.append(mesh.get_surface_area())

            # Get the bounding box dimensions and center
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_center = bbox.get_center()
            bbox_dims = bbox.get_max_bound() - bbox.get_min_bound()

            # Move the object so that the bounding box center is above the origin and the 
            # bottom of the bounding box is at z=0
            mesh.translate(-bbox_center)
            mesh.translate([0, 0, bbox_dims[2] / 2])

            mesh.compute_vertex_normals()
            obj_meshes.append(mesh)

        obj_surface_areas = np.array(obj_surface_areas)
        obj_counts = np.array(obj_counts)

        total_area = np.sum(obj_surface_areas * obj_counts)
        obj_num_pts = ((obj_surface_areas / total_area) * self.num_query_points).astype(int)
        obj_num_pts[-1] += (self.num_query_points - np.sum(obj_num_pts * obj_counts))

        obj_point_clouds = []
        for i in range(len(obj_meshes)):
            if (obj_counts[i] == 1):
                pc = obj_meshes[i].sample_points_uniformly(obj_num_pts[i])
            else:
                pc = obj_meshes[i].sample_points_uniformly(int(self.SAMPLING_UPSCALE_FACTOR * obj_num_pts[i]))
            obj_point_clouds.append(pc)

        query_points = []
        query_normals = []
        query_class_labels = []
        query_instance_labels = []
        query_pt_colors = []

        for i in range(len(obj_classes)):
            for j in range(obj_counts[i]):

                # Define a random position and orientation for the object - no randomness in position currently
                z_rot = random.uniform(0, 360)
                rotation = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, z_rot])

                # Create transformation matrix
                transform = np.zeros((4, 4))
                transform[:3, :3] = rotation
                transform[3, 3] = 1

                # Apply transformation to the point cloud
                transformed_pc = obj_point_clouds[i].transform(transform)

                points = np.asarray(transformed_pc.points, dtype=np.float32)
                normals = np.asarray(transformed_pc.normals, dtype=np.float32)
                if obj_counts[i] > 1:
                    indices = np.random.choice(int(1.25 * obj_num_pts[i]), obj_num_pts[i], replace=False)
                    points = points[indices]
                    normals = normals[indices]
                
                query_points.append(points)
                query_normals.append(normals)
                if i == 0:
                    query_class_labels.append(np.ones(obj_num_pts[i], dtype=np.float32))
                else:
                    query_class_labels.append(np.zeros(obj_num_pts[i], dtype=np.float32))
                query_instance_labels.append(j * np.ones(obj_num_pts[i], dtype=np.float32))
                query_pt_colors.append(obj_colors[i].reshape(1, 3).repeat(obj_num_pts[i], axis=0))

        # Move objects into a grid-like pattern
        num_rows = int(np.sqrt(num_scene_instances - 1))
        num_cols = int((np.ceil(num_scene_instances - 1) / num_rows))
        spacing = self.plane_side_dim / (num_cols + 1)

        # Move the objects into a grid-like pattern with randomly sampled ordering for translation
        ordering = np.arange(num_scene_instances + 1)
        np.random.shuffle(ordering[1:])
        for i in range(num_scene_instances):
            order_idx = ordering[i]
            row = (order_idx - 1) % num_rows
            col = (order_idx - 1) // num_rows
            x = (col - num_cols / 2) * spacing
            y = (row - num_rows / 2) * spacing
            
            query_points[i] += np.array([x, y, 0])

        # Create sample data
        query_points = np.concatenate(query_points, axis=0)
        query_normals = np.concatenate(query_normals, axis=0)
        query_pc = torch.from_numpy(np.concatenate([query_points, query_normals], axis=-1))
        class_labels = torch.from_numpy(np.concatenate(query_class_labels, axis=0))
        instance_labels = torch.from_numpy(np.concatenate(query_instance_labels, axis=0))

        # Create support point cloud
        support_pc = obj_meshes[0].sample_points_uniformly(self.num_support_points)
        support_pc = np.concatenate([np.asarray(support_pc.points), np.asarray(support_pc.normals)], axis=-1)
        support_pc = torch.from_numpy(support_pc).to(torch.float32)

        # Return query point cloud, support point cloud, labels, and instance labels as dictionary
        data = {
            "query": query_pc[:self.num_query_points],
            "class_labels": class_labels[:self.num_query_points],
            "instance_label": instance_labels[:self.num_query_points],
            "support": support_pc
        }
        
        if self.debug_mode is True:
            data["colors"] = np.concatenate(query_pt_colors, axis=0)[:self.num_query_points]
            data["obj_classes"] = obj_classes
            data["obj_counts"] = obj_counts

        return data
    
    def __len__(self) -> int:
        return self.dataset_size


if __name__ == "__main__":
    # # Create a dataset object
    # dataset = FindAnythingDataset(debug_mode=True)

    # # Visualize one point cloud from the dataset
    # data = dataset[0]
    # print("Total number of points (including plane):", len(data["query"]))
    # for obj, obj_count in zip(data["obj_classes"], data["obj_counts"]):
    #     print(f"class: {obj:<15} \t count: {obj_count:02d}")

    # # Create an open3D visualization window
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([1, 1, 1])

    # point_cloud = data['query'].numpy()

    # scene_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud[:, :3]))
    # scene_pc.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    # scene_pc.colors = o3d.utility.Vector3dVector(data["colors"])
    
    # vis.add_geometry(scene_pc)
    # vis.run()
    # vis.destroy_window()

    import time
    import torch
    from torch.utils.data import DataLoader

    dataset = FindAnythingDataset(split="train")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=8,
    )

    start = time.time()
    for _ in dataloader:
        print(time.time() - start)
        start = time.time() 
