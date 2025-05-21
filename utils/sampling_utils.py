import torch, os
import torchvision


def max_weights_sampling(scene, render, gaussians, pipe, bg, model_path, iteration):
    train_camera_stack = scene.getTrainCameras().copy()
    pixels_error_stack = []
    max_weights_stack = []
    mask_stack = []
    for train_camera in train_camera_stack:
        render_pkg = render(train_camera, gaussians, pipe, bg)
        image, max_weights_idx = render_pkg["render"], render_pkg["max_weights_idx"][0]
        gt_image = train_camera.original_image.cuda()
        l1_pixels = torch.abs((image - gt_image)).mean(0).detach()
        pixels_error_stack.append(l1_pixels)
        max_weights_stack.append(max_weights_idx)
        mask_stack.append(l1_pixels>torch.quantile(l1_pixels, 0.99))
        a = image.clone()
        a[:, l1_pixels>torch.quantile(l1_pixels, 0.99)] = torch.tensor([[1.], [0.], [0.]]).cuda()
        os.makedirs(os.path.join(model_path, "error", "{}".format(train_camera.image_name)), exist_ok=True)
        torchvision.utils.save_image(a, os.path.join(model_path, "error", "{}".format(train_camera.image_name), '{}_{}'.format(iteration, train_camera.image_name) + ".png"))
        
    pixels_error_stack = torch.stack(pixels_error_stack, dim=0)
    max_weights_stack = torch.stack(max_weights_stack, dim=0)
    mask_stack = torch.stack(mask_stack, dim=0)
    
    return max_weights_stack, mask_stack