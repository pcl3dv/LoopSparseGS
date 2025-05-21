import os
import numpy as np
import sys
import sqlite3
from plyfile import PlyData, PlyElement
import imageio
from argparse import ArgumentParser
import warnings

warnings.filterwarnings('ignore')

IS_PYTHON3 = sys.version_info[0] >= 3
MAX_IMAGE_ID = 2**31 - 1

if sys.platform.startswith('win'):
    REMOVE = 'rmdir /s /q '
    COPY = 'copy '
else:
    REMOVE = 'rm -r '
    COPY = 'cp -r '

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

def round_python3(number):
    rounded = round(number)
    if abs(number - rounded) == 0.5:
        return 2.0 * round(number / 2.0)
    return rounded

def init_colmap(scene, base_path, n_views):
    llffhold = 8
    view_path = str(n_views) + '_views'
    os.chdir(os.path.join(base_path, scene))
    
    os.system(REMOVE + view_path)
    os.mkdir(view_path)
    os.chdir(view_path)
    os.mkdir('created')
    os.mkdir('triangulated')
    os.mkdir('images')
    os.system('colmap model_converter  --input_path ../sparse/0/ --output_path ../sparse/0/  --output_type TXT') # 转换bin文件为txt

    images = {}
    with open('../sparse/0/images.txt', "r") as fid:
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

    img_list = sorted(images.keys(), key=lambda x: x)
    train_img_list = [c for idx, c in enumerate(img_list) if idx % llffhold != 0]
    if n_views > 0:
        idx_sub = [round_python3(i) for i in np.linspace(0, len(train_img_list)-1, n_views)]
        train_img_list = [c for idx, c in enumerate(train_img_list) if idx in idx_sub]


    for img_name in train_img_list:
        os.system(COPY + r'..\images\\' + img_name + r'  images\\' + img_name)

    os.system(COPY + r'..\sparse\0\cameras.txt created\.')
    with open(r'created\points3D.txt', "w") as fid:
        pass

    res = os.popen( 'colmap feature_extractor --database_path database.db --image_path images  --SiftExtraction.max_image_size 4032 --SiftExtraction.max_num_features 32768 --SiftExtraction.estimate_affine_shape 1 --SiftExtraction.domain_size_pooling 1').read()
    os.system( 'colmap exhaustive_matcher --database_path database.db --SiftMatching.guided_matching 1 --SiftMatching.max_num_matches 32768')
    db = COLMAPDatabase.connect('database.db')
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    print(img_rank, res)
    with open('created/images.txt', "w") as fid:
        for idx, img_name in enumerate(img_rank):
            print(img_name)
            data = [str(1 + idx)] + [' ' + item for item in images[os.path.basename(img_name)]] + ['\n\n']
            fid.writelines(data)

    os.system('colmap point_triangulator --database_path database.db --image_path images --input_path created  --output_path triangulated  --Mapper.ba_local_max_num_iterations 40 --Mapper.ba_local_max_refinements 3 --Mapper.ba_global_max_num_iterations 100')
    os.system('colmap model_converter  --input_path triangulated --output_path triangulated  --output_type TXT')


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

def init_depth(source_path, train_sub, reso=8):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    # make train idx
    database_path = os.path.join(source_path, "{}_views/database.db".format(train_sub))
    db = COLMAPDatabase.connect(database_path)
    db_images = db.execute("SELECT * FROM images")
    img_rank = [db_image[1] for db_image in db_images]
    train_idx = []
    images_txt = os.path.join(source_path, "{}_views/created/images.txt".format(train_sub))
    for idx, img_name in enumerate(img_rank):
        print(img_name)
        if 'pseudo' not in img_name:
            train_idx.append(1 + idx)
    
    
    
    points_path = os.path.join(source_path, '{}_views/triangulated/points3D.txt'.format(train_sub))
    camera_path = os.path.join(source_path, '{}_views/created/cameras.txt'.format(train_sub))
    images_path = os.path.join(source_path, '{}_views/created/images.txt'.format(train_sub))
    depth_save_file = os.path.join(source_path, '{}_views/depth_{}'.format(train_sub, reso))
    os.makedirs(depth_save_file, exist_ok=True)
    
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


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
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
                if len(uses_train_images) > 0:
                    xyz = np.array(tuple(map(float, elems[1:4])))
                    rgb = np.array(tuple(map(int, elems[4:7])))
                    error = np.array(float(elems[7]))
                    xyzs[count] = xyz
                    rgbs[count] = rgb
                    errors[count] = error
                    count += 1

    xyzs = xyzs[:count]
    rgbs = rgbs[:count]
    errors = errors[:count]
    
    storePly(os.path.join(source_path, '{}_views/triangulated/points3D.ply'.format(train_sub)), xyzs, rgbs)
    
    with open(camera_path, 'r+') as fcam:
        line = fcam.readlines()[-1].strip().split(' ')
        W = int(line[2])//reso
        H = int(line[3])//reso
        focal = float(line[4])/reso
    
    # points to depth map
    with open(images_path, 'r+') as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip().split(' ')
            if len(line) > 1:
                depth_images = np.zeros([H, W])
                
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
                
                rays_d = (xyzs-C2W[:3, 3]) @ (W2C[:3, :3].T ) # camera coord
                points_image_plane = rays_d/np.abs(rays_d[:, 2][:, None])
                ref_pixels = np.stack((points_image_plane[:, 0]*focal+W/2, -(points_image_plane[:, 1] * focal) + H/2, ), axis=1)
                ref_pixels = np.int32(ref_pixels)
                
                in_pixel_mask = ((ref_pixels[:, 0] >= 0) & (ref_pixels[:, 0] < W) & \
                                (ref_pixels[:, 1] >= 0) & (ref_pixels[:, 1] < H)) & (ref_pixels[:, -1] >= 1)
                
                ref_pixels = ref_pixels[in_pixel_mask]
                depth_images[ref_pixels[:,1], ref_pixels[:,0]] = rays_d[in_pixel_mask, 2]/-1

                imageio.imwrite(os.path.join(depth_save_file, line[-1][:-3]+'png'), np.round(depth_images).astype(np.uint16))
                
    return xyzs, rgbs, errors


if __name__ == '__main__':
    parser = ArgumentParser(description="Data Prepare")
    parser.add_argument("--source_path", "-s", default='./data/nerf_llff_data', type=str)
    parser.add_argument("--train_sub", default=3, type=int)
    parser.add_argument("--skip_colmap", action="store_true")
    parser.add_argument("--skip_depth", action="store_true")
    args = parser.parse_args()
    
    source_path, train_sub = args.source_path, args.train_sub
    raw_path = os.getcwd()
    for scene in ['fern', 'flower', 'fortress',  'horns',  'leaves',  'orchids',  'room',  'trex']:
        # init n_view images and colmap
        if not args.skip_colmap:
            init_colmap(scene, source_path, train_sub)  # please use absolute path!
            os.chdir(raw_path)
        
        # init n_view depth maps.
        if not args.skip_depth:
            init_depth(os.path.join(source_path, scene), train_sub, reso=8)
