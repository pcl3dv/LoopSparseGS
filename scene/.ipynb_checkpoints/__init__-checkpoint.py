#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.pose_utils import generate_pseudo_poses
from scene.cameras import PseudoCamera


class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, \
                     shuffle=True, resolution_scales=[1.0], dataset_type=None, train_sub=-1, PMS_init=False, debug_init=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians


        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.pseudo_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, \
                                                          dataset_type=dataset_type, train_sub=train_sub, \
                                                          pseudo_loop_iters=args.pseudo_loop_iters, PMS_init=PMS_init, debug_init=debug_init)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            # load pseudo view
            if load_iteration == -1: args.pseudo_loop_iters+=1 # for rendering
            if args.use_pseudo_view and args.pseudo_loop_iters>0:
                print("Loading Test Cameras")
                pseudo_view_poses_path = os.path.join(args.source_path, 'pseudo_poses.npy')
                if os.path.exists(pseudo_view_poses_path):
                    pseudo_poses = np.load(pseudo_view_poses_path)
                else:
                    pseudo_poses = generate_pseudo_poses(self.train_cameras[resolution_scale], dataset_type)
                    np.save(pseudo_view_poses_path, pseudo_poses)
                pseudo_loop_iters = args.pseudo_loop_iters
                pseudo_poses = pseudo_poses[ : pseudo_loop_iters * args.loop_image_nums]
                
                # pseudo_path
                pseudo_render_file = os.path.join(args.model_path, 'pseudo', "loop_{}".format(pseudo_loop_iters))
                pseudo_depth_file = os.path.join(args.source_path, '{}_views_{}/depth_8'.format(train_sub, pseudo_loop_iters))
                
                # TODO using a json file to save the relation between the index of pseudo view poses and pseudo image/depth names.
                # pseudo camera views
                view = self.train_cameras[resolution_scale].copy()[0]
                self.pseudo_cameras[resolution_scale] = [PseudoCamera(
                    R=pose[:3, :3].T, T=pose[:3, 3], FoVx=view.FoVx, FoVy=view.FoVy,
                    width=view.image_width, height=view.image_height, 
                    depth_path=os.path.join(pseudo_depth_file, 'pseudo{:03d}.png'.format(idx)), 
                    rgb_path=os.path.join(pseudo_depth_file, 'pseudo{:03d}_color.png'.format(idx)), 
                    render_path=os.path.join(pseudo_render_file, 'pseudo_img{:03d}.png'.format(idx)),
                ) for idx, pose in enumerate(pseudo_poses)]
            else:
                self.pseudo_cameras[resolution_scale] = []
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getPseudoCameras(self, scale=1.0):
        return self.pseudo_cameras[scale]