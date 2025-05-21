import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_pearson_loss, compute_patch_pearson_loss, compute_depth_loss
from lpipsPyTorch import lpips
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
import torch.nn.functional as F
from arguments import ModelParams, PipelineParams, OptimizationParams
# from torchmetrics.functional.regression import spearman_corrcoef
from torchmetrics.functional.regression import pearson_corrcoef
from utils.general_utils import vis_depth, modify
from utils.depth_utils import estimate_depth
from utils.sampling_utils import max_weights_sampling
import torchvision
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
depth2img = lambda x: ((x-x.min())/(x.max()-x.min()))
def training(dataset, opt, pipe, args):
    
    testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, dataset_type, train_sub, PMS_init = args.test_iterations, \
        args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.dataset_type, args.train_sub, args.PMS_init

    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    dataset.model_path = args.model_path
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, dataset_type=dataset_type, train_sub=train_sub, PMS_init=PMS_init, debug_init=args.debug_init)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    record_psnr = {
        'psnr_process': [],
        'max_psnr': 0,
        'max_ssim': 0,
        'max_lpips': 0,
        'max_iter' : -1,
        }
    
    print("\nTraining Start! Switch depth l1 loss: {}, mono pearson loss: {}, \nPseudo supervised:{}, pseudo_loop_iters:{}, sparse_sampling:{}".format(args.use_depth, \
                                            args.use_pearson, args.use_pseudo_view, args.pseudo_loop_iters, args.sparse_sampling))
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % args.increase_SH_iter == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, iteration=-1)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if args.sparse_sampling and iteration > args.start_opacity_iter:
            opacity_mask = torch.ones(gaussians.get_scaling.size(0), dtype=bool)
            opacity_mask[render_pkg['max_weights_idx'].unique().long()] = False
            opacity_loss = args.opacity_weight * gaussians.get_opacity[opacity_mask].mean()
            loss += opacity_loss
        else:
            opacity_loss = torch.tensor(0.0, device='cuda')
        
        # depth loss
        if args.use_depth:
            # compute train-view L1 depth loss.
            colmap_depth = torch.from_numpy(viewpoint_cam.depth_image).cuda()
            depth_loss = compute_depth_loss(colmap_depth, render_pkg['depth'][0])
            loss += args.depth_weight * depth_loss
                
            # compute train-view pearson loss.
            if args.use_pearson and args.patch_length == -1 and iteration > args.start_pearson_iter:
                mono_depth = torch.from_numpy(viewpoint_cam.mono_depth_image).cuda()
                pearson_loss = compute_pearson_loss(mono_depth, render_pkg['depth'][0])
            elif args.use_pearson and iteration > args.start_pearson_iter:
                mono_depth = torch.from_numpy(viewpoint_cam.mono_depth_image).cuda()
                pearson_loss = compute_patch_pearson_loss(mono_depth, render_pkg['depth'][0], p_l=args.patch_length)
            else:
                pearson_loss = torch.tensor(0.0, device='cuda')
            loss += args.pearson_weight * pearson_loss
            
            
            if args.use_pseudo_view and args.pseudo_loop_iters>0:
                # add l1 loss
                if not pseudo_stack:
                    pseudo_stack = scene.getPseudoCameras().copy()
                pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
                render_pkg_pseudo = render(pseudo_cam, gaussians, pipe, background)
                
                # compute pseudo-view L1 depth loss.
                colmap_depth_pseudo = torch.from_numpy(pseudo_cam.depth_image).cuda()
                depth_loss_pseudo = compute_depth_loss(colmap_depth_pseudo, render_pkg_pseudo['depth'][0])
                loss += args.depth_weight * depth_loss_pseudo
                
                # compute pseudo-view pearson loss.
                if args.use_pearson and args.patch_length == -1 and iteration > args.start_pearson_iter:
                    mono_depth_pseudo = estimate_depth(render_pkg_pseudo['render'], mode='train')
                    pearson_loss_pseudo = compute_pearson_loss(mono_depth_pseudo, render_pkg_pseudo['depth'][0])
                elif args.use_pearson and iteration > args.start_pearson_iter:
                    mono_depth_pseudo = estimate_depth(render_pkg_pseudo['render'], mode='train')
                    pearson_loss_pseudo = compute_patch_pearson_loss(mono_depth_pseudo, render_pkg_pseudo['depth'][0], p_l=args.patch_length)
                else:
                    pearson_loss_pseudo = torch.tensor(0.0, device='cuda')
                loss += args.pearson_weight * pearson_loss_pseudo
                
                # compute pseudo-view color loss.
                if args.use_pseudo_color:
                    # valid_depth_pseudo_mask.reshape(1, pseudo_gt_image.shape[1:])
                    pseudo_gt_image = pseudo_cam.rgb_image
                    valid_depth_pseudo_mask = colmap_depth_pseudo > 0.
                    pseudo_color_mask = valid_depth_pseudo_mask.reshape(1, pseudo_gt_image.shape[1], pseudo_gt_image.shape[2]).repeat(3, 1, 1)
                    pseudo_rgb_loss = l1_loss(render_pkg_pseudo['render'][pseudo_color_mask], pseudo_gt_image[pseudo_color_mask])
                    loss += (1.0 - opt.lambda_dssim) * pseudo_rgb_loss
                else:
                    pseudo_rgb_loss = torch.tensor(0.0, device='cuda')
        else:
            depth_loss = torch.tensor(0.0, device='cuda')
            pearson_loss = torch.tensor(0.0, device='cuda')
            
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % (10) == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # adding max psnr record
            record_psnr = training_report(tb_writer, iteration, Ll1, pearson_loss, depth_loss, loss, l1_loss, opacity_loss, iter_start.elapsed_time(iter_end), \
                                          testing_iterations, scene, render, (pipe, background), record_psnr=record_psnr)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = None
                    # update according pixels_error
                    if args.sparse_sampling:
                        max_weights_stack, mask_stack = max_weights_sampling(scene, render, gaussians, pipe, bg, args.model_path, iteration)
                        weights_mask = max_weights_stack[mask_stack]
                    else:
                        weights_mask = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, weights_mask)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                
    # record middle training metrics.
    with open(os.path.join(args.model_path, 'record_psnr.txt'), "w+") as rf:
        rf.writelines(record_psnr['psnr_process'])
        rf.write("\n[ITER {}] max_PSNR {:.3f} max_SSIM {:.3f} max_LPIPS {:.3f}".format(record_psnr['max_iter'], \
                                                                                round(record_psnr['max_psnr'].cpu().item(), 3), \
                                                                                round(record_psnr['max_ssim'].cpu().item(), 3), \
                                                                                round(record_psnr['max_lpips'].cpu().item(), 3)))
        
def prepare_output_and_logger(args):    
    args.model_path = os.path.join("./output/", args.exp_name) if not args.model_path else os.path.join(args.model_path, args.exp_name)
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(os.path.join(args.model_path, "events"))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, pearson_loss, depth_loss, loss, l1_loss, opacity_loss, \
                    elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, record_psnr={}):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/opacity_loss', opacity_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/pearson_loss', pearson_loss.item(), iteration)
        
        tb_writer.add_scalar('train_loss_patches/depth_loss', depth_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})
        psnr_process = record_psnr['psnr_process']
        max_psnr, max_ssim, max_lpips, max_iter = record_psnr['max_psnr'], record_psnr['max_ssim'], record_psnr['max_lpips'], record_psnr['max_iter']
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    vis_render = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(vis_render["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    
                    if args.use_depth:
                        depth_map = depth2img(vis_render['depth'])[None]
                        if config['name'] == 'train':
                            gt_depth = depth2img(torch.from_numpy(viewpoint.depth_image).to('cuda'))[None, None]
                        else:
                            gt_depth = torch.zeros(depth_map.size(), device='cuda')
                        if tb_writer and (idx < 3):
                            tb_writer.add_images(config['name'] + "_view_{}/depth_render".format(viewpoint.image_name), depth_map, global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(config['name'] + "_view_{}/depth_ground_truth".format(viewpoint.image_name), gt_depth, global_step=iteration)
                        
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += lpips(image, gt_image, net_type='vgg').mean().double()
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                
                if config['name'] == "test":
                    print("\n[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.3f} SSIM {:.3f} LPIPS {:.3f}".format(iteration, config['name'], \
                                                                                         l1_test, psnr_test, ssim_test, lpips_test))
                    psnr_process.append("\n[ITER {}] Evaluating {}: L1 {:.4f} PSNR {:.3f} SSIM {:.3f} LPIPS {:.3f}".format(iteration, config['name'], \
                                                                                         l1_test, psnr_test, ssim_test, lpips_test))
                    if max_psnr < psnr_test:
                        max_psnr = psnr_test
                        max_ssim = ssim_test
                        max_lpips = lpips_test
                        max_iter = iteration
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()
        return {
            'psnr_process': psnr_process,
            'max_psnr': max_psnr,
            'max_ssim': max_ssim,
            'max_lpips': max_lpips,
            'max_iter': max_iter,
            }
    return record_psnr

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[i for i in range(1000, 10001, 1000)])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--dataset_type", type=str, default = 'llff')
    parser.add_argument("--train_sub", type=int, default = 3)  
    parser.add_argument("--debug_init", action="store_true", help="full images init, but few views train") 
    parser.add_argument("--PMS_init", "-d", action="store_true", help="patch match stereo densify points")
    parser.add_argument("--sparse_sampling", "-sps",action="store_true", help="open sparse-frienly densification")
    parser.add_argument("--patch_length", type=int, default = 32) 
    parser.add_argument("--increase_SH_iter", default=1000, help="patch match stereo densify points")
    parser.add_argument("--epoch_number", type=int, default = -1, help="hyper-parameters need consider the n_views epoches.")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # modify hyper-parameters according epoch_number, eg. epoch_number=100, new_iter = iter/100*3.
    if args.epoch_number != -1:
        args = modify(args)

    training(lp.extract(args), op.extract(args), pp.extract(args), args)
    # All done
    print("\nTraining complete.")
    os.system(f"python render.py -m {args.model_path} --skip_train --render_depth")
    # os.system(f"python tools/loop.py -p {args.pseudo_loop_iters} -s {args.source_path} -m {args.model_path} --train_sub {args.train_sub}")
        
        