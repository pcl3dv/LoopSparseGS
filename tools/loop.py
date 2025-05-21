# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:49:55 2024

@author: dell
"""

import numpy as np
import shutil
import os
from argparse import ArgumentParser

from plyfile import PlyData, PlyElement
import sqlite3
import sys
from colmap_depth import read_array, bin2depth
import warnings
import imageio
import gc

warnings.filterwarnings('ignore')

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    prior_qw REAL,
    prior_qx REAL,
    prior_qy REAL,
    prior_qz REAL,
    prior_tx REAL,
    prior_ty REAL,
    prior_tz REAL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(MAX_IMAGE_ID)

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = \
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"

CREATE_ALL = "; ".join([
    CREATE_CAMERAS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_KEYPOINTS_TABLE,
    CREATE_DESCRIPTORS_TABLE,
    CREATE_MATCHES_TABLE,
    CREATE_TWO_VIEW_GEOMETRIES_TABLE,
    CREATE_NAME_INDEX
])

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)

def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)

class COLMAPDatabase(sqlite3.Connection):

    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = \
            lambda: self.executescript(CREATE_CAMERAS_TABLE)
        self.create_descriptors_table = \
            lambda: self.executescript(CREATE_DESCRIPTORS_TABLE)
        self.create_images_table = \
            lambda: self.executescript(CREATE_IMAGES_TABLE)
        self.create_two_view_geometries_table = \
            lambda: self.executescript(CREATE_TWO_VIEW_GEOMETRIES_TABLE)
        self.create_keypoints_table = \
            lambda: self.executescript(CREATE_KEYPOINTS_TABLE)
        self.create_matches_table = \
            lambda: self.executescript(CREATE_MATCHES_TABLE)
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def update_camera(self, model, width, height, params, camera_id):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
            (model, width, height, array_to_blob(params),camera_id))
        return cursor.lastrowid

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def produce_idx2image_dict(images_path, source_path, data_type='llff'):
    idx2image_dict = {}
    image_dict = {}
    with open(images_path, 'r+') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip().split(' ')
            if len(line) > 1:
                idx2image_dict[line[0]] = line[-1]
                if 'pseudo' not in line[-1] and data_type != 'llff':
                    image_dict[line[0]] = imageio.imread(os.path.join(source_path, 'images', line[-1]))
                
    if data_type == 'llff':
        import json
        with open(os.path.join(source_path, 'name_index.json'), 'r+') as f_name:
            name_index_json = json.load(f_name)
        for k, v in idx2image_dict.items():
            if 'pseudo' not in v:
                idx2image_dict[k] = name_index_json[v]
                image_dict[k] = imageio.imread(os.path.join(source_path, 'images_8', name_index_json[v]))
        
    return idx2image_dict, image_dict
         
    
def produce_points_filter(xyzs, depth_dict, imgs_info, errors, image_plane_thres):
    # filter_1: border bbox filter
    min_range = np.percentile(xyzs, 0.1, axis=0)
    max_range = np.percentile(xyzs, 99.9, axis=0)
    filter_1_min = (xyzs[:, 0] > min_range[0]) & (xyzs[:, 1] > min_range[1]) & (xyzs[:, 2] > min_range[2])
    filter_1_max = (xyzs[:, 0] < max_range[0]) & (xyzs[:, 1] < max_range[1]) & (xyzs[:, 2] < max_range[2])
    filter_1 = filter_1_min & filter_1_max
    
    # filter_2: image_plane_filter
    filter_2 = np.ones_like(xyzs[:, 0], dtype=bool) 
    for i in depth_dict.keys():
        img_info = list(filter(lambda x:x.strip().split(' ')[0]==str(i), imgs_info))[0]
        img_info = img_info.strip().split(' ')
    
        qvec = np.array(tuple(map(float, img_info[1:5])))
        tvec = np.array(tuple(map(float, img_info[5:8])))
        R = qvec2rotmat(qvec)
        T = np.array(tvec)
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R
        Rt[:3, 3] = T
        Rt[3, 3] = 1.0
        C2W = np.linalg.inv(Rt)
        C2W = np.concatenate([C2W[:, 0:1], -C2W[:, 1:2], -C2W[:, 2:3], C2W[:, 3:]], 1)
        W2C = np.linalg.inv(C2W)
        
        points_xyzs_image = (xyzs-C2W[:3, 3]) @ (W2C[:3, :3].T)
        image_plan_filter = points_xyzs_image[:, 2]<image_plane_thres
        filter_2 = filter_2 & image_plan_filter
    
    # filter_3: errors mask
    filter_3 = (errors < 2)[:, 0]
    
    return filter_1, filter_2, filter_3

def mask_strategy_1(f_mask_1, points_path, num_points, depth_dict):
    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    points_trainview_corr = []
    with open(points_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                idx = np.array(tuple(map(int, elems[8::2])))
                uses_train_images = np.intersect1d(idx, train_idx)
                if f_mask_1:
                    if len(uses_train_images) > 0:
                        xyz = np.array(tuple(map(float, elems[1:4])))
                        rgb = np.array(tuple(map(int, elems[4:7])))
                        error = np.array(float(elems[7]))
                        xyzs[count] = xyz
                        rgbs[count] = rgb
                        errors[count] = error
                        points_trainview_corr.append(uses_train_images[0])  # 每个points对应的train_view_id, 用于图像像素匹配。
                        
                        for _i in np.unique(idx): # Q: Why one 3D point has multiple 2D points in one image?
                            depth_dict[str(_i)].append(count)
                        count += 1
                else:
                    xyz = np.array(tuple(map(float, elems[1:4])))
                    rgb = np.array(tuple(map(int, elems[4:7])))
                    error = np.array(float(elems[7]))
                    xyzs[count] = xyz
                    rgbs[count] = rgb
                    errors[count] = error
                    for _i in np.unique(idx): # Q: Why one 3D point has multiple 2D points in one image?
                        depth_dict[str(_i)].append(count)
                    count += 1
                    
    return xyzs, rgbs, errors, count, points_trainview_corr, depth_dict

def read_points3D_text(source_path, train_sub, pseudo_loop_iters, train_idx, depth_dict, data_type, image_plane_thres=-12, f_mask_1=True, f_mask_2=True, f_mask_3=True):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points_path = os.path.join(source_path, '{}_views_{}/triangulated/points3D.txt'.format(train_sub, pseudo_loop_iters))
    camera_path = os.path.join(source_path, '{}_views_{}/created/cameras.txt'.format(train_sub, pseudo_loop_iters))
    images_path = os.path.join(source_path, '{}_views_{}/created/images.txt'.format(train_sub, pseudo_loop_iters))
    depth_save_file = os.path.join(source_path, '{}_views_{}/depth_8'.format(train_sub, pseudo_loop_iters))
    os.makedirs(depth_save_file, exist_ok=True)
    idx2image_dict, image_dict = produce_idx2image_dict(images_path, source_path, data_type)
    
    
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(points_path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    with open(images_path, 'r+') as f:
        a = f.readlines()

    xyzs, rgbs, errors, count, points_trainview_corr, depth_dict = mask_strategy_1(f_mask_1, points_path, num_points, depth_dict)

    xyzs = xyzs[:count]
    rgbs = rgbs[:count]
    errors = errors[:count]
    
    with open(camera_path, 'r+') as fcam:
        line = fcam.readlines()[-1].strip().split(' ')
        W = int(line[2])//8
        H = int(line[3])//8
        focal = float(line[4])/8
    
    # points filter
    # 1. Points with an error of one thousandth are filtered out.
    
    # make points diffuse color.
    with open(images_path, 'r+') as fp:
        imgs_info = fp.readlines()

        # filter 1:
    filter_1, filter_2, filter_3 = produce_points_filter(xyzs, depth_dict, imgs_info, errors, image_plane_thres)
    
    # filter_mask = filter_1 & filter_2
    filter_mask = filter_3
    # xyzs_mask = xyzs[filter_mask]
    # rgbs_mask = rgbs[filter_mask]
    # errors_mask = errors[filter_mask]
    xyzs_mask = xyzs
    rgbs_mask = rgbs
    errors_mask = errors
    storePly(os.path.join(source_path, '{}_views_{}/triangulated/points3D.ply'.format(train_sub, pseudo_loop_iters)), xyzs_mask, rgbs_mask)
    
    if f_mask_3:
        valid_index = np.where(filter_3)[0]
    else:
        valid_index = np.where(np.ones_like(filter_3))[0]
    # points to depth map
    with open(images_path, 'r+') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip().split(' ')
            if len(line) > 1:
                depth_images = np.zeros([H, W])
                color_images = np.zeros([H, W, 3])
                
                qvec = np.array(tuple(map(float, line[1:5])))
                tvec = np.array(tuple(map(float, line[5:8])))
                R = qvec2rotmat(qvec)
                T = np.array(tvec)
                
                Rt = np.zeros((4, 4))
                Rt[:3, :3] = R
                Rt[:3, 3] = T
                Rt[3, 3] = 1.0
                C2W = np.linalg.inv(Rt)
                C2W = np.concatenate([C2W[:, 0:1], -C2W[:, 1:2], -C2W[:, 2:3], C2W[:, 3:]], 1)
                W2C = np.linalg.inv(C2W)
                
                if f_mask_2:
                    image_xyzs = xyzs[np.intersect1d(depth_dict[line[0]], valid_index)]
                else:
                    image_xyzs = xyzs[valid_index]
                
                
                rays_d = (image_xyzs-C2W[:3, 3]) @ (W2C[:3, :3].T ) # camera coord
                points_image_plane = rays_d/np.abs(rays_d[:, 2][:, None])
                ref_pixels = np.stack((points_image_plane[:, 0]*focal+W/2, -(points_image_plane[:, 1] * focal) + H/2, ), axis=1)
                ref_pixels = np.int32(ref_pixels)
                
                in_pixel_mask = ((ref_pixels[:, 0] >= 0) & (ref_pixels[:, 0] < W) & \
                                (ref_pixels[:, 1] >= 0) & (ref_pixels[:, 1] < H)) & (ref_pixels[:, -1] >= 1)
                
                ref_pixels = ref_pixels[in_pixel_mask]
                depth_images[ref_pixels[:,1], ref_pixels[:,0]] = rays_d[in_pixel_mask, 2]/-1
                
                imageio.imwrite(os.path.join(depth_save_file, line[-1][:-3]+'png'), np.round(depth_images).astype(np.uint16))
                
    return xyzs, rgbs, errors


def make_imagestxt(pseudo_loop_iters, source_path, model_path, train_sub, loop_image_nums):
    source_dir = os.path.join(source_path, "{}_views".format(train_sub))
    target_dir = os.path.join(source_path, "{}_views_{}".format(train_sub, pseudo_loop_iters))
    os.makedirs(target_dir, exist_ok=True)

    if not os.path.exists(os.path.join(target_dir, 'created')):
        shutil.copytree(os.path.join(source_dir, 'created'), os.path.join(target_dir, 'created'))
    if not os.path.exists(os.path.join(target_dir, 'images')):
        shutil.copytree(os.path.join(source_dir, 'images'), os.path.join(target_dir, 'images'))

    pseudo_poses = np.load(os.path.join(source_path, 'pseudo_poses.npy'))
    
    with open(os.path.join(source_dir, 'created/images.txt'), 'r+') as f1:
        context = f1.readlines()
        
    for i in range(loop_image_nums * pseudo_loop_iters):
        # copy rendering images
        image_name = 'pseudo{:03d}.png'.format(i)
        shutil.copy(os.path.join(model_path, "pseudo/loop_{}/{}".format(pseudo_loop_iters, image_name)),
                                 os.path.join(target_dir, 'images/{}'.format(image_name)))
        
        # remake image.txt

        pose_i = pseudo_poses[i]
        qvec = rotmat2qvec(pose_i[:3, :3])
        tvec = pose_i[:3, 3]
        context.append('{} {} {} {} {} {} {} {} 1 {}\n'.format(i+1+train_sub, qvec[0], qvec[1], qvec[2], qvec[3], \
                                                                    tvec[0], tvec[1], tvec[2], image_name))
        context.append('\n')
    images = {}
    with open(os.path.join(target_dir, 'created/images.txt'), 'w+') as f2:
        f2.writelines(context)
    
    # saved dict, {'images_name: context in images.txt'}
    with open(os.path.join(target_dir, 'created/images.txt'), "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                fid.readline().split()
                images[image_name] = elems[1:]
    return images


def make_fused_ply(pseudo_loop_iters, source_path, model_path, train_sub, images):
    database_path = os.path.join(source_path, "{}_views_{}/database.db".format(train_sub, pseudo_loop_iters))
    images_path = os.path.join(source_path, "{}_views_{}/images".format(train_sub, pseudo_loop_iters))
    camera_path = os.path.join(source_path, "{}_views_{}/created/cameras.txt".format(train_sub, pseudo_loop_iters))
    
    res = os.system('colmap feature_extractor --database_path {} --image_path {} --ImageReader.camera_model SIMPLE_PINHOLE --ImageReader.single_camera 1 --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1'.format(database_path, images_path))
    gc.collect()
    os.system('colmap exhaustive_matcher --database_path {} --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768'.format(database_path))
    gc.collect()
    # set existing intr parameters.
    os.system('python ./tools/intr.py --database_path {} --camera_txt_path {}'.format(database_path, camera_path))
    gc.collect()
    # modify the order of images name in images.txt to same as the database.
    db = COLMAPDatabase.connect(database_path)
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    print(img_rank, res)
    train_idx = []
    depth_dict = {}
    images_txt = os.path.join(source_path, "{}_views_{}/created/images.txt".format(train_sub, pseudo_loop_iters))
    with open(images_txt, "w") as fid:
        for idx, img_name in enumerate(img_rank):
            print(img_name)
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            fid.writelines(data)
            depth_dict[str(1 + idx)] = []
            if 'pseudo' not in img_name:
                train_idx.append(1 + idx)
    
    # sparse reconstruction
    sparse_input_path = os.path.join(source_path, "{}_views_{}/created".format(train_sub, pseudo_loop_iters))
    sparse_output_path = os.path.join(source_path, "{}_views_{}/triangulated".format(train_sub, pseudo_loop_iters))
    os.makedirs(sparse_output_path, exist_ok=True)
    os.system('colmap point_triangulator --database_path {} --image_path {} --input_path {} --output_path {} --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100 --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0'.format(database_path, images_path, sparse_input_path, sparse_output_path))
    gc.collect()
    os.system('colmap model_converter  --input_path {} --output_path {}  --output_type TXT'.format(sparse_output_path, sparse_output_path))
    
    
    
    return train_idx, depth_dict

def make_depth_maps(train_sub, pseudo_loop_iters, train_idx):
    depth_path = os.path.join(source_path, "{}_views_{}/depth".format(train_sub, pseudo_loop_iters))
    depth_path_8 = os.path.join(source_path, "{}_views_{}/depth_8".format(train_sub, pseudo_loop_iters))
    os.makedirs(depth_path, exist_ok=True)
    os.makedirs(depth_path_8, exist_ok=True)

    fB = 32504; 
    min_depth_percentile = 2
    max_depth_percentile = 98

    depthmapsdir = os.path.join(source_path, "{}_views_{}/dense/stereo/depth_maps".format(train_sub, pseudo_loop_iters))
    depthmap_name = list(filter(lambda x:'geometric' in x, os.listdir(depthmapsdir)))
    # output_dir = os.path.join(source_path, r'3_views\depth')
    # output_dir_8 = os.path.join(database, scene_name, 'depth_8')

    for depthmap in depthmap_name:
        imgname = depthmap.split('.')[0]
        depthmap = os.path.join(depthmapsdir, depthmap)
        bin2depth(imgname, depthmap, depth_path, depth_path_8)



if __name__ == '__main__':
    parser = ArgumentParser(description="Looping colmap parameters")
    parser.add_argument("--pseudo_loop_iters", "-p", default=0, type=int)
    parser.add_argument("--source_path", "-s", default='../data/nerf_llff_data/fern', type=str)
    parser.add_argument("--data_type", default='llff', type=str)
    parser.add_argument("--model_path", "-m", default='../output/fern', type=str)
    parser.add_argument("--train_sub", default=3, type=int)
    parser.add_argument("--image_plane_thres", default=-12, type=float)
    
    args = parser.parse_args()

    # make image.txt
    pseudo_loop_iters = args.pseudo_loop_iters + 1
    train_sub = args.train_sub
    source_path = args.source_path
    model_path = args.model_path
    loop_image_nums = train_sub*4
    data_type = args.data_type
    image_plane_thres = args.image_plane_thres
    
    images = make_imagestxt(pseudo_loop_iters=pseudo_loop_iters, source_path=source_path, model_path=model_path, train_sub=train_sub, loop_image_nums=loop_image_nums)
    print('successful create images.txt')
    
    train_idx, depth_dict = make_fused_ply(pseudo_loop_iters=pseudo_loop_iters, source_path=source_path, model_path=model_path, train_sub=train_sub, images=images)
    print('successful create fused.ply')
    
    read_points3D_text(source_path, train_sub, pseudo_loop_iters, train_idx, depth_dict, data_type, image_plane_thres = image_plane_thres)
    print('successful create depth maps')




