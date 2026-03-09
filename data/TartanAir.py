import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
from data.transforms import FlowAugmentor

import os
import math
import random
import h5py
import cv2
from tqdm import tqdm
from glob import glob
import os.path as osp
from utils import frame_utils

import sys

sys.path.append(os.getcwd())

K = np.array([[320, 0, 320],
              [0, 320, 240],
              [0, 0, 1]], dtype=np.float32)

"""
Code from https://github.com/huyaoyu/ImageFlow/blob/master/ImageFlow.py
"""

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        # self.subsample_groundtruth = False
        if aug_params is not None:
            self.augmentor = FlowAugmentor(**aug_params)

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.extra_info = []

    def __getitem__(self, index):
        while True:
            try:
                return self.fetch(index)
            except Exception as e:
                index = random.randint(0, len(self) - 1)
            return self.fetch(index)

    def read_flow(self, index):
        raise NotImplementedError

    def fetch(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        index = index % len(self.image_list)
        flow, valid = self.read_flow(index)
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[..., None], (1, 1, 3))
            img2 = np.tile(img2[..., None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        valid = torch.from_numpy(valid)
        valid = (valid >= 0.5) & ((~torch.isnan(flow)).all(dim=0)) & ((~torch.isinf(flow)).all(dim=0))
        flow[torch.isinf(flow)] = 0
        flow[torch.isnan(flow)] = 0
        return img1, img2, flow, valid.float()

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self

    def __len__(self):
        return len(self.image_list)


def from_homog(x):
    return x[...,:-1] / x[...,[-1]]

def transform(T, p):
    assert T.shape == (4,4)
    return np.einsum('H W j, i j -> H W i', p, T[:3,:3]) + T[:3, 3]

def reproject(depth1, pose1, pose2, K1, K2):
    H, W = depth1.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    img_1_coords = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float64)
    cam1_coords = np.einsum('H W, H W j, i j -> H W i', depth1, img_1_coords, np.linalg.inv(K1))
    rel_pose = pose2 @ np.linalg.inv(pose1)
    cam2_coords = transform(rel_pose, cam1_coords)
    return from_homog(np.einsum('H W j, i j -> H W i', cam2_coords, K2))

def induced_flow(depth0, depth1, data):
    H, W = depth0.shape
    coords1 = reproject(depth0, data['T0'], data['T1'], data['K0'], data['K1'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_01 = coords1 - coords0
    H, W = depth1.shape
    coords1 = reproject(depth1, data['T1'], data['T0'], data['K1'], data['K0'])
    x, y = np.meshgrid(np.arange(W), np.arange(H), indexing='xy')
    coords0 = np.stack([x, y], axis=-1)
    flow_10 = coords1 - coords0
    return flow_01, flow_10

def check_cycle_consistency(flow_01, flow_10):
    H, W = flow_01.shape[:2]
    new_coords = flow_01 + np.stack(
        np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), axis=-1
    )
    flow_reprojected = cv2.remap(
        flow_10, new_coords.astype(np.float32), None, interpolation=cv2.INTER_LINEAR
    )
    cycle = flow_reprojected + flow_01
    cycle = np.linalg.norm(cycle, axis=-1)
    mask = (cycle < 0.1 * min(H, W)).astype(np.float32)
    return mask

def from_quaternion_to_rotation_matrix(q):
    """
    q: A numpy vector, 4x1.
    """
    qi2 = q[0, 0] ** 2
    qj2 = q[1, 0] ** 2
    qk2 = q[2, 0] ** 2
    qij = q[0, 0] * q[1, 0]
    qjk = q[1, 0] * q[2, 0]
    qki = q[2, 0] * q[0, 0]
    qri = q[3, 0] * q[0, 0]
    qrj = q[3, 0] * q[1, 0]
    qrk = q[3, 0] * q[2, 0]
    s = 1.0 / (q[3, 0] ** 2 + qi2 + qj2 + qk2)
    ss = 2 * s
    R = [ \
        [1.0 - ss * (qj2 + qk2), ss * (qij - qrk), ss * (qki + qrj)], \
        [ss * (qij + qrk), 1.0 - ss * (qi2 + qk2), ss * (qjk - qri)], \
        [ss * (qki - qrj), ss * (qjk + qri), 1.0 - ss * (qi2 + qj2)], \
        ]
    R = np.array(R, dtype=np.float32)
    return R


class TartanAir(FlowDataset):
    def __init__(self, aug_params=None, root='/home/johndoe/data-1/jogndoe/Optical/datasets/TartanAir'):
        super(TartanAir, self).__init__(aug_params)
        self.root = root
        self.worldT = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.build_dataset_adjacent()

    def build_dataset_adjacent(self):
        scenes = glob(osp.join(self.root, '*/*/*/'))
        for scene in sorted(scenes):
            images = sorted(glob(osp.join(scene, 'image_left/*.png')))
            for idx in range(len(images) - 1):
                frame0 = str(idx).zfill(6)
                frame1 = str(idx + 1).zfill(6)
                self.image_list.append([images[idx], images[idx + 1]])
                self.flow_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_flow.npy"))
                self.mask_list.append(osp.join(scene, 'flow', f"{frame0}_{frame1}_mask.npy"))

    def process_tartanair_pose(self, data):
        data = data.reshape((-1, 1))
        t = data[:3, 0].reshape((-1, 1))
        q = data[3:, 0].reshape((-1, 1))
        R = from_quaternion_to_rotation_matrix(q)
        T = np.eye(4)
        T[:3, :3] = R.transpose()
        T[:3, 3] = -R.transpose().dot(t).reshape((-1,))
        T = self.worldT @ T
        return T

    def build_dataset_all_pair(self):
        self.depth_list = []
        self.cam_list = []
        scenes = glob(osp.join(self.root, '*/*/*/'))
        for scene in sorted(scenes):
            for view in ['left']:
                images = sorted(glob(osp.join(scene, f"image_{view}/*.png")))
                depths = sorted(glob(osp.join(scene, f"depth_{view}/*.npy")))
                tartanair_pose_data = np.loadtxt(osp.join(scene, f"pose_{view}.txt"))
                poses = [self.process_tartanair_pose(data) for data in tartanair_pose_data]
                for i in range(len(images) - 1):
                    for j in range(i + 1, min(len(images), i + self.n_frames + 1)):
                        self.image_list.append([images[i], images[j]])
                        self.depth_list.append([depths[i], depths[j]])
                        self.cam_list.append([poses[i], poses[j]])

    def read_flow_adjacent(self, index):
        flow = np.load(self.flow_list[index])
        valid = np.load(self.mask_list[index])
        # rescale the valid mask to [0, 1]
        valid = 1 - valid / 100
        return flow, valid

    def read_flow_all_pair(self, index):
        T0 = self.cam_list[index][0]
        T1 = self.cam_list[index][1]
        depth0 = np.load(self.depth_list[index][0])
        depth1 = np.load(self.depth_list[index][1])
        cam_data = {'T0': T0, 'T1': T1, 'K0': K, 'K1': K}
        flow_01, flow_10 = induced_flow(depth0, depth1, cam_data)
        valid_01 = check_cycle_consistency(flow_01, flow_10)
        flow_01[valid_01 == 0] = 0
        return flow_01, valid_01

    def read_flow(self, index):
        return self.read_flow_adjacent(index)