import os
import random
import matplotlib
import numpy as np
import open3d as o3d

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
            dataset_size: int = 1e6,
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
        self.object_size_range = [2, 3]
        self.plane_side_dim = 12

        self.cmap = matplotlib.colormaps['jet']

        # Get list of all objects separated by train and test per class
        self.train_obj_dict = dict()
        self.test_obj_dict = dict()
        for obj_class in self.obj_classes:
            self.train_obj_dict[obj_class] = [obj for obj in os.listdir(
                os.path.join(root_dir, obj_class, "train")) if obj.endswith(".off")]
            self.test_obj_dict[obj_class] = [obj for obj in os.listdir(
                os.path.join(root_dir, obj_class, "test")) if obj.endswith(".off")]
            
    def check_collision(self, bbox1, bbox2):
        """Check if two bounding boxes are colliding in x-y plane"""
        min_1 = bbox1.get_min_bound()
        max_1 = bbox1.get_max_bound()
        min_2 = bbox2.get_min_bound()
        max_2 = bbox2.get_max_bound()

        return not ((max_1[0] < min_2[0] or min_1[0] > max_2[0]) or \
            (max_1[1] < min_2[1] or min_1[1] > max_2[1]))
    
    def create_plane(self, side_dim):
        # Generate vertices of the plane centered at (0,0,0)
        half_side = side_dim / 2.0
        vertices = np.array([[-half_side, -half_side, 0],
                            [half_side, -half_side, 0],
                            [half_side, half_side, 0],
                            [-half_side, half_side, 0]])
        # Generate triangular faces of the plane
        triangles = np.array([[0, 1, 2], [0, 2, 3]])
        # Create an Open3D TriangleMesh object
        plane = o3d.geometry.TriangleMesh()
        plane.vertices = o3d.utility.Vector3dVector(vertices)
        plane.triangles = o3d.utility.Vector3iVector(triangles)
        plane.compute_vertex_normals()
        return plane
    
    def random_positions(self, side_dim, obj_point_clouds, obj_counts, discretization_factor = 0.1):
        # Create a 0.1 discretized boolean 2D array with the same side as the side_dim initialized to 1's
        grid = np.ones((int(side_dim / discretization_factor), 
                        int(side_dim / discretization_factor)), 
                        dtype=bool)

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
                max_obj_side_dim = max(obj_point_clouds[i].get_max_bound() - obj_point_clouds[i].get_min_bound())
                offset = int(np.ceil(max_obj_side_dim * 0.75 / discretization_factor))
                min_dim0 = max(0, sampled_index[0] - offset)
                max_min0 = min(grid.shape[0], sampled_index[0] + offset)
                min_dim1 = max(0, sampled_index[1] - offset)
                max_dim1 = min(grid.shape[1], sampled_index[1] + offset)
                grid[min_dim0:max_min0, min_dim1:max_dim1] = False

                # Calculate actual 2D position
                position = np.array([sampled_index[0] * 0.1 - side_dim / 2.0, 
                                    sampled_index[1] * 0.1 - side_dim / 2.0, 
                                    0])
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
        num_scene_instances = np.random.randint(self.min_scene_instances, self.max_scene_instances)

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
        obj_colors = [np.array(self.cmap(i))[:3] for i in np.linspace(0, 1, num=len(obj_classes) + 1)]
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
            mesh_bbox = mesh.get_axis_aligned_bounding_box()
            mesh.scale(scale / np.max(mesh_bbox.get_max_bound() - mesh_bbox.get_min_bound()), center=mesh.get_center())
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

        # Create a ground plane
        plane = self.create_plane(self.plane_side_dim)
        obj_classes.append('plane')
        obj_class_ids.append(0)
        obj_counts.append(1)
        obj_meshes.append(plane)
        obj_surface_areas.append(self.plane_side_dim**2)

        obj_surface_areas = np.array(obj_surface_areas)
        obj_counts = np.array(obj_counts)

        total_area = np.sum(obj_surface_areas * obj_counts)
        obj_num_pts = ((obj_surface_areas / total_area) * self.num_query_points).astype(int)
        # Add residual points from integer division to last object
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
                if i != len(obj_classes) - 1:
                    # Define a random orientation about the z-axis
                    z_rot = random.uniform(0, 360)
                    rotation = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, z_rot])

                    # Create transformation matrix
                    transform = np.zeros((4, 4))
                    transform[:3, :3] = rotation
                    transform[3, 3] = 1

                    # Apply transformation to the point cloud
                    transformed_pc = obj_point_clouds[i].transform(transform)
                else:
                    transformed_pc = obj_point_clouds[i]

                points = np.asarray(transformed_pc.points, dtype=np.float32)
                normals = np.asarray(transformed_pc.normals, dtype=np.float32)

                # If plane, set all normals to pointing straight up
                if i == len(obj_classes) - 1:
                    normals = np.zeros_like(normals)
                    normals[:, 2] = 1

                if obj_counts[i] > 1:
                    indices = np.random.choice(int(self.SAMPLING_UPSCALE_FACTOR * obj_num_pts[i]), obj_num_pts[i], replace=False)
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

        # Randomly sample positions for all objects
        half_side_with_offset = self.plane_side_dim - self.object_size_range[1]
        random_positions = self.random_positions(half_side_with_offset, obj_point_clouds, obj_counts)
        for i in range(len(query_points) - 1):
            query_points[i] += random_positions[i]

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
    # Create a dataset object
    dataset = FindAnythingDataset(debug_mode=True)

    # Visualize one point cloud from the dataset
    data = dataset[0]
    print("Total number of points (including plane):", len(data["query"]))
    for obj, obj_count in zip(data["obj_classes"], data["obj_counts"]):
        print(f"class: {obj:<15} \t count: {obj_count:02d}")

    ### Visualize with Open3D ###

    # # Create an open3D visualization window
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    # opt.background_color = np.asarray([1, 1, 1])

    # scene_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud[:, :3]))
    # scene_pc.normals = o3d.utility.Vector3dVector(point_cloud[:, 3:])
    # scene_pc.colors = o3d.utility.Vector3dVector(data["colors"])

    # vis.add_geometry(scene_pc)
    # vis.run()
    # vis.destroy_window()

    ### Visualize with matplotlib ###

    # # Visualize the point cloud with matplotlib without Axes3D
    # point_cloud = data['query'].numpy()
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1)
    # plt.show()

    ### Test time ###
    import time
    import torch
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = FindAnythingDataset(split="train")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        num_workers=16,
    )

    start = time.time()
    for data in tqdm(dataloader, total=len(dataloader) // 16):
        print(time.time() - start)
        start = time.time()
