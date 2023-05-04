import os
import torch
import imageio
import pytorch3d
import numpy as np

from pathlib import Path
from typing import Optional, Tuple

import pytorch3d.transforms as t3d
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)


def get_points_renderer(
    image_size: int = 512,
    device: Optional[str] = None,
    radius: float = 0.01,
    background_color: Tuple = (1, 1, 1)
) -> PointsRenderer:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
    )
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(raster_settings=raster_settings),
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer


def get_mesh_renderer(image_size: int = 512, device: Optional[str] = None) -> MeshRenderer:
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device),
    )

    return renderer


@torch.no_grad()
def render_mesh(
    mesh: Meshes,
    output_file: Path,
    image_size: int = 400,
    camera_distance: float = 15,
    rotate_mesh: Optional[Tuple] = [np.pi/2, 0, 0],
    num_frames: int = 60,
) -> None:
    device = mesh.device

    if rotate_mesh is not None:
        rotation = t3d.euler_angles_to_matrix(torch.tensor(rotate_mesh), "XYZ")
        rotate = t3d.Rotate(rotation)
        mesh = Meshes(
            verts=rotate.transform_points(mesh.verts_padded()),
            faces=mesh.faces_padded(),
            textures=mesh.textures
        )

    if mesh.textures is None:
        alpha = torch.ones_like(mesh.verts_padded()[...,-1:])
        color = alpha * torch.tensor([0, 0, 1], device=device, dtype=alpha.dtype)
        mesh.textures = pytorch3d.renderer.TexturesVertex(color)

    renderer = get_mesh_renderer(image_size=image_size, device=device)

    images = []
    for azim in np.arange(-180, 180, 360 / num_frames):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=camera_distance,
            elev=45,
            azim=azim,
        )

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, fov=60, device=device)
        lights = pytorch3d.renderer.PointLights(location=(-R @ T[..., None])[..., 0], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        image = rend.cpu().numpy()[0, ..., :3] * 255  # (B, H, W, 4) -> (H, W, 3)
        images.append(image.astype(np.uint8))

    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)
    imageio.mimsave(output_file, images, format='GIF', loop=0, duration=4)


@torch.no_grad()
def render_point_cloud(
    points: Pointclouds,
    output_file: Path,
    colors: Optional[torch.Tensor] = None,
    rotate_scene: Optional[Tuple] = [np.pi/2, 0, 0],
    camera_distance: float = 15,
    image_size: int = 400,
    num_frames: int = 60,
) -> None:
    device = points.device

    if colors is None:
        alpha = torch.ones_like(points[...,-1:])
        colors = alpha * torch.tensor([0, 0, 1], device=device, dtype=alpha.dtype)
    
    if rotate_scene is not None:
        rotation = t3d.euler_angles_to_matrix(torch.tensor(rotate_scene), "XYZ")
        rotate = t3d.Rotate(rotation)
        points = rotate.transform_points(points)

    point_cloud = pytorch3d.structures.Pointclouds(
        points=points.to(device),
        features=colors.to(device),
    )

    render = get_points_renderer(image_size=image_size, device=device, radius=0.01)

    images = []
    for azim in np.arange(-180, 180, 360 / num_frames):
        R, T = pytorch3d.renderer.cameras.look_at_view_transform(
            dist=camera_distance,
            elev=45,
            azim=azim,
        )

        camera = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = pytorch3d.renderer.PointLights(location=(-R @ T[..., None])[..., 0], device=device)

        render_output = render(point_cloud, cameras=camera, lights=lights)
        image = 255 * render_output[0, ..., :3].cpu().numpy()
        images.append(image.astype(np.uint8))

    if not os.path.exists(output_file.parent):
        os.makedirs(output_file.parent)
    imageio.mimsave(output_file, images, format='GIF', loop=0, duration=4)
