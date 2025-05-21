import argparse
import numpy as np
import os
import struct
from PIL import Image
import warnings
import os
import cv2, imageio

warnings.filterwarnings('ignore') # 屏蔽nan与min_depth比较时产生的警告

fB = 32504;
min_depth_percentile = 2
max_depth_percentile = 98

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def bin2depth(i, depth_map, depthdir, depthdir_8):
    # depth_map = '0.png.geometric.bin'
    # print(depthdir)
    # if min_depth_percentile > max_depth_percentile:
    #     raise ValueError("min_depth_percentile should be less than or equal "
    #                      "to the max_depth_perceintile.")

    # Read depth and normal maps corresponding to the same image.
    assert os.path.exists(depth_map)

    depth_map = read_array(depth_map)

    min_depth, max_depth = np.percentile(depth_map[depth_map>0], [min_depth_percentile, max_depth_percentile])
    depth_map[depth_map <= 0] = np.nan # 把0和负数都设置为nan，防止被min_depth取代
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    depth_map_2 = np.nan_to_num(depth_map)

    # np.save(os.path.join(depthdir_8, i+'.npy'), depth_map)  
    depth_8 = np.round(depth_map_2).astype(np.uint16)
    depth_8 = cv2.resize(depth_8, (504, 378), interpolation=cv2.INTER_NEAREST)
    imageio.imwrite(os.path.join(depthdir_8, i+'.png'), depth_8)
    
    
    
    depth_map_3 = np.nan_to_num(depth_map)
    depth_map_3 = np.round(depth_map_3).astype(np.uint16)
    imageio.imwrite(os.path.join(depthdir, i+'.png'), depth_map_3)
    
    maxdisp = fB / min_depth;
    mindisp = fB / max_depth;
    depth_map = (fB/depth_map - mindisp) * 255 / (maxdisp - mindisp);
    depth_map = np.nan_to_num(depth_map) # nan全都变为0
    depth_map = depth_map.astype(int)

    image = Image.fromarray(depth_map).convert('L')
    image.save(os.path.join(depthdir, i+'_visual.png'))

    # image_8 = image.resize((504, 378), Image.LANCZOS) # 保证resize为 504 x 378
    # image_8.save(os.path.join(depthdir_8, i+'.JPG'))

if __name__ == '__main__':
    scene_list = ['fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex']
    n_views = 3
    database = r"G:\Sp3DGS\exp1_depth\data\nerf_llff_data"
    

    
    for scene_name in scene_list:
        depthmapsdir = os.path.join(database, scene_name, r'3_views\dense\stereo\depth_maps')
        depthmap_name = list(filter(lambda x:'geometric' in x, os.listdir(depthmapsdir)))
        output_dir = os.path.join(database, scene_name, r'3_views\depth')
        output_dir_8 = os.path.join(database, scene_name, 'depth_8')
        
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        if not os.path.exists(output_dir_8): os.mkdir(output_dir_8)
        
        for depthmap in depthmap_name:
            imgname = depthmap.split('.')[0]
            depthmap = os.path.join(depthmapsdir, depthmap)
            bin2depth(imgname, depthmap, output_dir, output_dir_8)
        
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
