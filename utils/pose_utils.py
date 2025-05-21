import numpy as np
from typing import Tuple



def normalize(x):
    return x / np.linalg.norm(x)


def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def viewmatrix(lookdir, up, position, subtract_position=False):
  """Construct lookat view matrix."""
  vec2 = normalize((lookdir - position) if subtract_position else lookdir)
  vec0 = normalize(np.cross(up, vec2))
  vec1 = normalize(np.cross(vec2, vec0))
  m = np.stack([vec0, vec1, vec2, position], axis=1)
  return m


def poses_avg(poses):
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)
  z_axis = poses[:, :3, 2].mean(0)
  up = poses[:, :3, 1].mean(0)
  cam2world = viewmatrix(z_axis, up, position)
  return cam2world


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = poses_avg(poses) # average pos, average z-axis.
  transform = np.linalg.inv(pad_poses(cam2world))
  poses = transform @ pad_poses(poses)
  return unpad_poses(poses), transform


def generate_llff_pseudo_poses(train_cameras, loop_iters_all=12, num_virtual_per_epoch=4, initial_spread=0.02, expansion_rate=0.1):
    "generate pseudo poses for llff dataset"
    n_poses = 10000
    poses, bounds = [], []
    for camera in train_cameras:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([camera.R.T, camera.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view) # colmap c2w, y down, z forward
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        bounds.append(camera.bounds)
    poses = np.stack(poses, 0)
    bounds = np.stack(bounds)
    
    scale = 1. / (bounds.min() * .75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    poses, transform = recenter_poses(poses)
    
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / ( ((1-dt)/close_depth) + (dt/inf_depth) )
    
    # Get radii for spiral path using 90th percentile of camera positions
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])
    
    # Generate random poses.
    
    random_poses_all = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    
    buffer_ratio = (np.max(positions, axis=0) - np.min(positions, axis=0))*0.1
    min_bounds = np.min(positions, axis=0) - buffer_ratio
    max_bounds = np.max(positions, axis=0) + buffer_ratio
    
    
    for epoch in range(0, loop_iters_all):
        random_poses_epoch = []
        spread = initial_spread + expansion_rate * epoch
        for real_cam in positions:
            for _ in range(num_virtual_per_epoch):
                valid_camera = False
                while not valid_camera:
                    virtual_cam = np.random.normal(loc=real_cam, scale=spread, size=(3,))
                    if np.all(virtual_cam >= min_bounds) and np.all(virtual_cam <= max_bounds):
                        valid_camera = True
                        position = cam2world @ np.concatenate([virtual_cam, [1,]])
                        lookat = cam2world @ [0, 0, -focal, 1.]
                        z_axis = position - lookat
                        random_pose = np.eye(4)
                        random_pose[:3] = viewmatrix(z_axis, up, position)
                        random_pose = np.linalg.inv(transform) @ random_pose
                        random_pose[:3, 1:3] *= -1
                        random_pose[:3, 3] /= scale
                        random_poses_epoch.append(np.linalg.inv(random_pose))
        random_poses_all.extend(random_poses_epoch)
    random_poses_all = np.stack(random_poses_all, axis=0)
    
    return random_poses_all
    

def generate_llff_pseudo_poses_raw(train_cameras):
    "generate pseudo poses for llff dataset"
    n_poses = 10000
    poses, bounds = [], []
    for camera in train_cameras:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([camera.R.T, camera.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view) # colmap c2w, y down, z forward
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
        bounds.append(camera.bounds)
    poses = np.stack(poses, 0)
    bounds = np.stack(bounds)
    
    scale = 1. / (bounds.min() * .75)
    poses[:, :3, 3] *= scale
    bounds *= scale
    poses, transform = recenter_poses(poses)
    
    # Find a reasonable 'focus depth' for this dataset as a weighted average
    # of near and far bounds in disparity space.
    close_depth, inf_depth = bounds.min() * .9, bounds.max() * 5.
    dt = .75
    focal = 1 / ( ((1-dt)/close_depth) + (dt/inf_depth) )
    
    # Get radii for spiral path using 90th percentile of camera positions
    positions = poses[:, :3, 3]
    radii = np.percentile(np.abs(positions), 100, 0)
    radii = np.concatenate([radii, [1.]])
    
    # Generate random poses.
    random_poses = []
    cam2world = poses_avg(poses)
    up = poses[:, :3, 1].mean(0)
    for _ in range(n_poses):
        t = radii * np.concatenate([2 * np.random.rand(3) - 1., [1,]])
        position = cam2world @ t
        lookat = cam2world @ [0, 0, -focal, 1.]
        z_axis = position - lookat
        random_pose = np.eye(4)
        random_pose[:3] = viewmatrix(z_axis, up, position)
        random_pose = np.linalg.inv(transform) @ random_pose
        random_pose[:3, 1:3] *= -1
        random_pose[:3, 3] /= scale
        random_poses.append(np.linalg.inv(random_pose))
    render_poses = np.stack(random_poses, axis=0)
    return render_poses

def generate_pseudo_poses(train_cameras, data_type):
    "generate pseudo poses according datatype!"
    if data_type == "llff":
        return generate_llff_pseudo_poses(train_cameras)
    

