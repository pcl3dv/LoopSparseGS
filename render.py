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

import torch
from scene import Scene
import os, sys
import numpy as np
import cv2
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
depth2img = lambda x: ((x-x.min())/(x.max()-x.min()))

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, render_depth):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if render_depth: makedirs(depth_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        result = render(view, gaussians, pipeline, background)
        rendering = result["render"]
        depth = depth2img(result["depth"])
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{}'.format(view.image_name) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{}'.format(view.image_name) + ".png"))
        if render_depth: torchvision.utils.save_image(depth, os.path.join(depth_path, '{}'.format(view.image_name) + ".png"))
        if render_depth: np.save(os.path.join(depth_path, '{}'.format(view.image_name) + ".npy"), result["depth"][0].cpu().numpy())
        
def render_pseudo(model_path, name, loop_iters, views, gaussians, pipeline, background, dataset_type):
    if dataset_type == 'llff':
        render_path = os.path.join(model_path, name, "loop_{}/renders".format(loop_iters))
        render_path_new = os.path.join(model_path, name, "loop_{}".format(loop_iters))
    else:
        render_path = os.path.join(model_path, name, "loop_{}".format(loop_iters))
        render_path_new = None
    
    makedirs(render_path, exist_ok=True)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, 'pseudo{0:03d}'.format(idx) + ".png"))
        
        if render_path_new is not None:
            rendering_new = rendering.permute(1,2,0).cpu().numpy()
            h, w = rendering_new.shape[0], rendering_new.shape[1]
            rendering_new = cv2.resize(rendering_new, (w*8, h*8), interpolation=cv2.INTER_AREA)
            rendering_new = torch.from_numpy(np.transpose(rendering_new, (2, 0, 1)))
            
            torchvision.utils.save_image(rendering_new, os.path.join(render_path_new, 'pseudo{0:03d}'.format(idx) + ".png"))
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_pseudo : bool, args):
    dataset_type, train_sub, PMS_init, debug_init, render_depth = args.dataset_type, args.train_sub, args.PMS_init, args.debug_init, args.render_depth
    
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, \
                      dataset_type=dataset_type, train_sub=train_sub, PMS_init=PMS_init, debug_init=debug_init)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, render_depth)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, render_depth)

        if not skip_pseudo:
             render_pseudo(dataset.model_path, "pseudo", dataset.pseudo_loop_iters, scene.getPseudoCameras(), gaussians, pipeline, background, dataset_type)

if __name__ == "__main__":
    # sys.argv.append('-m')
    # sys.argv.append('./output/fern_1')
    # sys.argv.append('--skip_train')
    # # sys.argv.append('--render_depth')
    # # sys.argv.append('--skip_test')
    # sys.argv.append('--skip_pseudo')

    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_pseudo", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.skip_pseudo, args)
    