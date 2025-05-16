# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# colmap mapper/colmap point_triangulator/glomap mapper from mast3r matches
# --------------------------------------------------------
import pycolmap
import torch
import pypose as pp
from torch import nn
from torch.nn import functional as F
import os
import os.path as path
import kapture.io
import kapture.io.csv
import subprocess
import PIL
from tqdm import tqdm
import PIL.Image
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path

from mast3r.model import AsymmetricMASt3R
from mast3r.colmap.database import export_matches, get_im_matches

import mast3r.utils.path_to_dust3r  # noqa
from dust3r_visloc.datasets.utils import get_resize_function

import kapture
from kapture.converter.colmap.database_extra import get_colmap_camera_ids_from_db, get_colmap_image_ids_from_db
from kapture.utils.paths import path_secure

from dust3r.datasets.utils.transforms import ImgNorm
from dust3r.inference import inference


def scene_prepare_images(root: str, maxdim: int, patch_size: int, image_paths: List[str]):
    images = []
    # image loading
    for idx in tqdm(range(len(image_paths))):
        rgb_image = PIL.Image.open(os.path.join(root, image_paths[idx])).convert('RGB')

        # resize images
        W, H = rgb_image.size
        resize_func, _, to_orig = get_resize_function(maxdim, patch_size, H, W)
        rgb_tensor = resize_func(ImgNorm(rgb_image))

        # image dictionary
        images.append({'img': rgb_tensor.unsqueeze(0),
                       'true_shape': np.int32([rgb_tensor.shape[1:]]),
                       'to_orig': to_orig,
                       'idx': idx,
                       'instance': image_paths[idx],
                       'orig_shape': np.int32([H, W])})
    return images


def remove_duplicates(images, image_pairs):
    pairs_added = set()
    pairs = []
    for (i, _), (j, _) in image_pairs:
        smallidx, bigidx = min(i, j), max(i, j)
        if (smallidx, bigidx) in pairs_added:
            continue
        pairs_added.add((smallidx, bigidx))
        pairs.append((images[i], images[j]))
    return pairs


def run_mast3r_matching(model: AsymmetricMASt3R, maxdim: int, patch_size: int, device,
                        kdata: kapture.Kapture, root_path: str, image_pairs_kapture: List[Tuple[str, str]],
                        colmap_db,
                        dense_matching: bool, pixel_tol: int, conf_thr: float, skip_geometric_verification: bool,
                        min_len_track: int, output_path: str):
    assert kdata.records_camera is not None
    image_paths = kdata.records_camera.data_list()
    image_path_to_idx = {image_path: idx for idx, image_path in enumerate(image_paths)}
    image_path_to_ts = {kdata.records_camera[ts, camid]: (ts, camid) for ts, camid in kdata.records_camera.key_pairs()}

    images = scene_prepare_images(root_path, maxdim, patch_size, image_paths)
    image_pairs = [((image_path_to_idx[image_path1], image_path1), (image_path_to_idx[image_path2], image_path2))
                   for image_path1, image_path2 in image_pairs_kapture]
    matching_pairs = remove_duplicates(images, image_pairs)

    colmap_camera_ids = get_colmap_camera_ids_from_db(colmap_db, kdata.records_camera)
    colmap_image_ids = get_colmap_image_ids_from_db(colmap_db)
    im_keypoints = {idx: {} for idx in range(len(image_paths))}

    im_matches = {}
    image_to_colmap = {}
    for image_path, idx in image_path_to_idx.items():
        _, camid = image_path_to_ts[image_path]
        colmap_camid = colmap_camera_ids[camid]
        colmap_imid = colmap_image_ids[image_path]
        image_to_colmap[idx] = {
            'colmap_imid': colmap_imid,
            'colmap_camid': colmap_camid
        }

    things_to_save = {
        'meta': {},
        'depth': {},
        'pose': {},
        'conf': {}
    }

    # compute 2D-2D matching from dust3r inference
    for chunk in tqdm(range(0, len(matching_pairs), 4)):
        pairs_chunk = matching_pairs[chunk:chunk + 4]
        output = inference(pairs_chunk, model, device, batch_size=1, verbose=False)
        pred1, pred2 = output['pred1'], output['pred2']
        # TODO handle caching
        im_images_chunk, raw_im_images_chunk = get_im_matches(pred1=pred1, pred2=pred2, pairs=pairs_chunk, image_to_colmap=image_to_colmap,
                                         im_keypoints=im_keypoints, conf_thr=conf_thr, is_sparse=not dense_matching,
                                         pixel_tol=pixel_tol)
        im_matches.update(im_images_chunk.items())

        # SAVE by waterch
        for pair_idx, pair in enumerate(pairs_chunk):
            idx1, idx2 = pair[0]['idx'], pair[1]['idx']
            if (idx1, idx2) in raw_im_images_chunk:
                if idx1 not in things_to_save['meta']:
                    things_to_save['meta'][idx1] = {'idx': idx1, 'instance': pair[0]['instance'], 'orig_shape': pair[0]['orig_shape']}
                if idx2 not in things_to_save['meta']:
                    things_to_save['meta'][idx2] = {'idx': idx2, 'instance': pair[1]['instance'], 'orig_shape': pair[1]['orig_shape']}
                geometry_from_pair(idx1, idx2, pred1['pts3d'][pair_idx], pred2['pts3d_in_other_view'][pair_idx], pred1['conf'][pair_idx], pred2['conf'][pair_idx], raw_im_images_chunk[(idx1, idx2)], things_to_save)
        # DONE

    merge_things_to_save(things_to_save)
    write_things(things_to_save, output_path)
    # filter matches, convert them and export keypoints and matches to colmap db
    colmap_image_pairs = export_matches(
        colmap_db, images, image_to_colmap, im_keypoints, im_matches, min_len_track, skip_geometric_verification)
    colmap_db.commit()

    return colmap_image_pairs


def pycolmap_run_triangulator(colmap_db_path, prior_recon_path, recon_path, image_root_path):
    print("running mapping")
    reconstruction = pycolmap.Reconstruction(prior_recon_path)
    pycolmap.triangulate_points(
        reconstruction=reconstruction,
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        refine_intrinsics=False,
    )


def pycolmap_run_mapper(colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    reconstructions = pycolmap.incremental_mapping(
        database_path=colmap_db_path,
        image_path=image_root_path,
        output_path=recon_path,
        options=pycolmap.IncrementalPipelineOptions({'multiple_models': False,
                                                     'extract_colors': True,
                                                     })
    )


def glomap_run_mapper(glomap_bin, colmap_db_path, recon_path, image_root_path):
    print("running mapping")
    args = [
        'mapper',
        '--database_path',
        colmap_db_path,
        '--image_path',
        image_root_path,
        '--output_path',
        recon_path
    ]
    args.insert(0, glomap_bin)
    glomap_process = subprocess.Popen(args)
    glomap_process.wait()

    if glomap_process.returncode != 0:
        raise ValueError(
            '\nSubprocess Error (Return code:'
            f' {glomap_process.returncode} )')


def kapture_import_image_folder_or_list(images_path: Union[str, Tuple[str, List[str]]], use_single_camera=False) -> kapture.Kapture:
    images = kapture.RecordsCamera()

    if isinstance(images_path, str):
        images_root = images_path
        file_list = [path.relpath(path.join(dirpath, filename), images_root)
                     for dirpath, dirs, filenames in os.walk(images_root)
                     for filename in filenames]
        file_list = sorted(file_list)
    else:
        images_root, file_list = images_path

    sensors = kapture.Sensors()
    for n, filename in enumerate(file_list):
        # test if file is a valid image
        try:
            # lazy load
            with PIL.Image.open(path.join(images_root, filename)) as im:
                width, height = im.size
                model_params = [width, height]
        except (OSError, PIL.UnidentifiedImageError):
            # It is not a valid image: skip it
            print(f'Skipping invalid image file {filename}')
            continue

        camera_id = f'sensor'
        if use_single_camera and camera_id not in sensors:
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)
        elif use_single_camera:
            assert sensors[camera_id].camera_params[0] == width and sensors[camera_id].camera_params[1] == height
        else:
            camera_id = camera_id + f'{n}'
            sensors[camera_id] = kapture.Camera(kapture.CameraType.UNKNOWN_CAMERA, model_params)

        images[(n, camera_id)] = path_secure(filename)  # don't forget windows

    return kapture.Kapture(sensors=sensors, records_camera=images)


from cv2 import estimateAffine3D
from collections import defaultdict, deque

def geometry_from_pair(idx1, idx2, pts3d1, pts3d2, conf1, conf2, matches, things_to_save):
    matched_pts1 = pts3d1[matches[0][:, 1], matches[0][:, 0]].cpu().numpy()
    matched_pts2 = pts3d2[matches[1][:, 1], matches[1][:, 0]].cpu().numpy()

    retval, out, inliers = estimateAffine3D(matched_pts1, matched_pts2)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :] = out

    proj = np.ones((pts3d2.shape[0] * pts3d2.shape[1], 4), dtype=np.float32)
    proj[:, :3] = pts3d2.reshape(-1, 3)
    proj = proj @ np.linalg.inv(pose).T
    proj = proj.reshape(pts3d2.shape[0], pts3d2.shape[1], 4)[..., :3]
    matched_proj = proj[matches[1][:, 1], matches[1][:, 0]]
    error = np.median(np.linalg.norm(matched_pts1 - matched_proj, axis=1))

    if idx1 not in things_to_save['depth']:
        things_to_save['depth'][idx1] = []
    things_to_save['depth'][idx1].append((pts3d1[..., 2], error))

    if not np.isnan(pose).any():
        things_to_save['pose'][(idx1, idx2)] = (pose, error)
    if error < 0.1:
        if idx2 not in things_to_save['depth']:
            things_to_save['depth'][idx2] = []
        things_to_save['depth'][idx2].append((proj[..., 2], error))
    
    if idx1 not in things_to_save['conf']:
        things_to_save['conf'][idx1] = []
    things_to_save['conf'][idx1].append((conf1, error))

    if idx2 not in things_to_save['conf']:
        things_to_save['conf'][idx2] = []
    things_to_save['conf'][idx2].append((conf2, error))

class PoseGraph(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.nodes = pp.Parameter(nodes)

    def forward(self, edges, poses):
        node1 = self.nodes[edges[..., 0]]
        node2 = self.nodes[edges[..., 1]]
        error = poses.Inv() @ node1.Inv() @ node2
        return error.Log().tensor()

    @staticmethod
    def build_graph(poses):
        nodes = []
        for pose in poses:
            nodes.append(pp.Pose(pose))
        return PoseGraph(nodes)

def merge_things_to_save(things_to_save):
    def merge_depth_maps():
        for idx in things_to_save['depth']:
            stacked = np.zeros((things_to_save['meta'][idx]['orig_shape'][0],
                                things_to_save['meta'][idx]['orig_shape'][1]), dtype=np.float32)
            errors = [e for d, e in things_to_save['depth'][idx]]
            errors = np.nan_to_num(errors, nan=1e6, posinf=1e6, neginf=1e6)  # 处理异常值
            scaled_errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors) + 1e-6)  # 归一化到[0,1]
            temperature = 0.1  # 温度系数平滑分布
            
            # 修改权重计算方式
            weights = torch.softmax(torch.from_numpy(-scaled_errors / temperature), 0).cpu().numpy()
            
            # 新增权重有效性检查
            valid_mask = np.isfinite(weights) & (weights > 1e-6)
            weights = np.where(valid_mask, weights, 0)
            weights /= weights.sum() + 1e-6

            for (d, e), w in zip(things_to_save['depth'][idx], weights):
                d = np.where(np.isfinite(d), d, 0.)
                stacked += F.interpolate(torch.from_numpy(d[None, None]), stacked.shape).squeeze().cpu().numpy() * w

            things_to_save['depth'][idx] = stacked

    def merge_conf_maps():
        for idx in things_to_save['conf']:
            stacked = np.zeros((things_to_save['meta'][idx]['orig_shape'][0],
                                things_to_save['meta'][idx]['orig_shape'][1]), dtype=np.float32)
            errors = [e for c, e in things_to_save['conf'][idx]]
            errors = np.nan_to_num(errors, nan=1e6, posinf=1e6, neginf=1e6)  # 处理异常值
            scaled_errors = (errors - np.min(errors)) / (np.max(errors) - np.min(errors) + 1e-6)  # 归一化到[0,1]
            temperature = 0.1  # 温度系数平滑分布
            
            # 修改权重计算方式
            weights = torch.softmax(torch.from_numpy(-scaled_errors / temperature), 0).cpu().numpy()
            
            # 新增权重有效性检查
            valid_mask = np.isfinite(weights) & (weights > 1e-6)
            weights = np.where(valid_mask, weights, 0)
            weights /= weights.sum() + 1e-6

            for (c, e), w in zip(things_to_save['conf'][idx], weights):
                c = np.where(np.isfinite(c), c, 0.)
                stacked += F.interpolate(torch.from_numpy(c[None, None]), stacked.shape).squeeze().cpu().numpy() * w

            things_to_save['conf'][idx] = stacked

    def build_minimum_spanning_tree():
        edges = []
        vertices = set()
        for (i,j), (_, err) in things_to_save['pose'].items():
            edges.append((i, j, err))
            vertices.update([i, j])

        parent = {v: v for v in vertices}
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]  # 路径压缩优化
                u = parent[u]
            return u

        edges.sort(key=lambda x: x[2])
        mst_edges = []
        for u, v, w in edges:
            if find(u) != find(v):
                mst_edges.append((u, v, w))
                parent[find(u)] = find(v)
        return mst_edges, vertices

    def compute_global_poses(mst_edges, vertices):
        things_to_save['global_pose'] = {0: np.eye(4)}
        
        # 构建邻接表
        adj = defaultdict(list)
        for u, v, w in mst_edges:
            adj[u].append(v)
            adj[v].append(u)

        # BFS遍历
        queue = deque([0])
        visited = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in adj[current]:
                if neighbor in visited:
                    continue
                
                # 获取相对位姿
                if (current, neighbor) in things_to_save['pose']:
                    rel_pose, _ = things_to_save['pose'][(current, neighbor)]
                else:
                    rel_pose, _ = things_to_save['pose'][(neighbor, current)]
                    rel_pose = np.linalg.inv(rel_pose)
                
                things_to_save['global_pose'][neighbor] = \
                    things_to_save['global_pose'][current] @ rel_pose
                queue.append(neighbor)

    # 主流程
    merge_depth_maps()
    merge_conf_maps()
    
    if not things_to_save['pose']:
        return

    mst_edges, vertices = build_minimum_spanning_tree()
    if mst_edges:
        things_to_save['mst_edges'] = mst_edges
        compute_global_poses(mst_edges, vertices)


def write_things(things_to_save, output_dir):
    from imageio import v3 as iio

    depth_dir = Path(output_dir) / 'depth'
    depth_dir.mkdir(exist_ok=True, parents=True)
    masks_dir = Path(output_dir) / 'masks'
    masks_dir.mkdir(exist_ok=True, parents=True)

    for idx, meta in things_to_save['meta'].items():
        instance = meta['instance']
        depth_instance = '.'.join(instance.split('.')[:-1]) + '.exr'
        mask_instance = '.'.join(instance.split('.')[:-1]) + '.png'
        depth = things_to_save['depth'][idx]
        # imwrite(str(depth_dir / instance), depth * 1e-3)
        iio.imwrite(str(depth_dir / depth_instance), depth)
        conf = things_to_save['conf'][idx]
        iio.imwrite(str(masks_dir / mask_instance), ((conf > conf.max() * 0.4) * 255).astype(np.uint8))
    pass