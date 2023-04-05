import argparse
# import glob
import json
import os
# from pathlib import Path

import cv2
# import matplotlib.pyplot as plt
import numpy as np
# import PIL
# from PIL import Image
# from torchvision import transforms



def form_transf(cam_rotmat, cam_transl):
    n_cams = cam_rotmat.shape[0]
    zeropadr = np.zeros((n_cams, 1,3))
    onepadt = np.ones((n_cams, 1,1))
    cam_transl = cam_transl.transpose([0,2,1]) # change to (b,3,1)
    cam_rotmat = np.concatenate([cam_rotmat, zeropadr],axis=1)
    cam_transl = np.concatenate([cam_transl, onepadt], axis=1)
    cam_transf = np.concatenate([cam_rotmat, cam_transl],axis=-1)
    
    return cam_transf



parser = argparse.ArgumentParser(description="preprocess zjumocap dataset to sdfstudio dataset")

parser.add_argument("--input_path", default='/mnt/hdd/datasets/ZJUMocap/CoreView_313')
parser.add_argument("--output_path", default='/mnt/hdd/datasets/ZJUMocap_SDFStudio/CoreView_313')
parser.add_argument("--timeidx", type=int, default=0)
parser.add_argument("--views", type=str, default='sparse')

parser.add_argument(
    "--type",
    dest="type",
    default="mono_prior",
    choices=["mono_prior", "sensor_depth"],
    help="mono_prior to use monocular prior, sensor_depth to use depth captured with a depth sensor (gt depth)",
)


args = parser.parse_args()



"""load and process camera"""
data_path = args.input_path
cam_data = np.load(os.path.join(data_path, 'mediapipe_all.pkl'), allow_pickle=True)
cam_rotmat_all = cam_data['cam_rotmat']
cam_transl_all = cam_data['cam_transl']
cam_K_all = cam_data['cam_K']
with open(os.path.join(data_path,'cam_params.json'),'r') as f:
    cam_names = json.load(f)['all_cam_names']

cam_names = np.array(cam_names)

cam_ids = np.arange(len(cam_names))
cam_ids_train = cam_ids[::2]
cam_ids_test = cam_ids[1:][::2] 

cam_rotmat = cam_rotmat_all[cam_ids]
cam_transl = cam_transl_all[cam_ids]
cam_K = cam_K_all[cam_ids]
        
poses = np.linalg.inv(form_transf(cam_rotmat, cam_transl)) # cam2world camera poses


"""process camera extrinsics"""
min_vertices = poses[:, :3, 3].min(axis=0)
max_vertices = poses[:, :3, 3].max(axis=0)

center = (min_vertices + max_vertices) / 2.0
scale = 2.0 / (np.max(max_vertices - min_vertices) + 3.0)
print(center, scale)

# we should normalize pose to unit cube
poses[:, :3, 3] -= center
poses[:, :3, 3] *= scale

# inverse normalization
scale_mat = np.eye(4).astype(np.float32)
scale_mat[:3, 3] -= center
scale_mat[:3] *= scale
scale_mat = np.linalg.inv(scale_mat)

"""process camera intrinsics"""
H, W = 1024, 1024
image_size=512

# center crop by 2 * image_size
offset_x = (W - image_size * 2) * 0.5
offset_y = (H - image_size * 2) * 0.5
cam_K[:,0, 2] -= offset_x
cam_K[:,1, 2] -= offset_y
# resize from 384*2 to 384
resize_factor = 0.5
cam_K[:,:2, :] *= resize_factor



"""copy and process images"""
output_path_train = os.path.join(args.output_path, 'train')
output_path_test = os.path.join(args.output_path, 'test')
input_path = args.input_path  # /mnt/hdd/datasets/ZJUMocap/CoreView_313
os.makedirs(output_path_train,exist_ok=True)
os.makedirs(output_path_test,exist_ok=True)



"""process train data"""
samplerate = 120
cam_names_train = cam_names[cam_ids_train]
mpresults = ['{}_mediapipe'.format(x) for x in cam_names_train]
tidx = 0
while tidx < 1000000:
    out_index = 0
    frames = []
    for mp in mpresults: # iteration over views
        img_path_mask = os.path.join(data_path, mp, 'frames_masked/image_{:05d}.png'.format(tidx))
        img_path = os.path.join(data_path, mp, 'frames_masked/image_{:05d}.png'.format(tidx))
        try:
            img = cv2.imread(img_path)
            img_mask = cv2.imread(img_path_mask)
        except:
            raise ValueError('all frames perhaps have been loaded.')
        
        img = cv2.resize(img,(512,512))
        img_mask = cv2.resize(img_mask,(512,512))
        img_mask = (img_mask>0).astype(int)*255
        outputpath = os.path.join(output_path_train, '{:06d}'.format(tidx))
        os.makedirs(outputpath, exist_ok=True)
        cv2.imwrite(os.path.join(outputpath,'{:06d}_rgb.png'.format(out_index)),img)
        cv2.imwrite(os.path.join(outputpath,'{:06d}_foreground_mask.png'.format(out_index)),img_mask)
        
        frame = {
        "rgb_path": '{:06d}_rgb.png'.format(out_index),
        "camtoworld": poses[cam_ids_train[out_index]].tolist(),
        "intrinsics": cam_K[cam_ids_train[out_index]].tolist(),
        "foreground_mask": '{:06d}_foreground_mask.png'.format(out_index),
        }
        frames.append(frame)
        out_index += 1


    # scene bbox for the zjumocap scene
    scene_box = {
        "aabb": [[-1, -1, -1], [1, 1, 1]],
        "near": 0.05,
        "far": 2.5,
        "radius": 1.0,
        "collider_type": "box",
    }

    # meta data
    output_data = {
        "camera_model": "OPENCV",
        "height": image_size,
        "width": image_size,
        "has_mono_prior": False,
        "has_sensor_depth": False,
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
    }

    output_data["frames"] = frames

    # save as json
    with open(os.path.join(outputpath, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


    tidx += samplerate




"""process test data"""
samplerate = 120
cam_names_test = cam_names[cam_ids_test]
mpresults = ['{}_mediapipe'.format(x) for x in cam_names_test]
tidx = 0
while tidx < 1000000:
    out_index = 0
    frames = []
    for mp in mpresults: # iteration over views
        img_path = os.path.join(data_path, mp, 'frames_masked/image_{:05d}.png'.format(tidx))
        try:
            img = cv2.imread(img_path)
        except:
            raise ValueError('all frames perhaps have been loaded.')
        
        img = cv2.resize(img,(512,512))
        img_mask = (img>0).astype(int)*255
        outputpath = os.path.join(output_path_test, '{:06d}'.format(tidx))
        os.makedirs(outputpath, exist_ok=True)
        cv2.imwrite(os.path.join(outputpath,'{:06d}_rgb.png'.format(out_index)),img)
        cv2.imwrite(os.path.join(outputpath,'{:06d}_foreground_mask.png'.format(out_index)),img_mask)
        
        frame = {
        "rgb_path": '{:06d}_rgb.png'.format(out_index),
        "camtoworld": poses[cam_ids_test[out_index]].tolist(),
        "intrinsics": cam_K[cam_ids_test[out_index]].tolist(),
        "foreground_mask": '{:06d}_foreground_mask.png'.format(out_index),
        }
        frames.append(frame)
        out_index += 1


    # scene bbox for the zjumocap scene
    scene_box = {
        "aabb": [[-1, -1, -1], [1, 1, 1]],
        "near": 0.05,
        "far": 2.5,
        "radius": 1.0,
        "collider_type": "box",
    }

    # meta data
    output_data = {
        "camera_model": "OPENCV",
        "height": image_size,
        "width": image_size,
        "has_mono_prior": False,
        "has_sensor_depth": False,
        "pairs": None,
        "worldtogt": scale_mat.tolist(),
        "scene_box": scene_box,
    }

    output_data["frames"] = frames

    # save as json
    with open(os.path.join(outputpath, "meta_data.json"), "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)


    tidx += samplerate

