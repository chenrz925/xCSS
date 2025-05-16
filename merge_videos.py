#!/usr/bin/env python3
import torch
import os
from os import path
import subprocess
import argparse
from itertools import combinations
from mast3r.model import AsymmetricMASt3R
from mast3r.retrieval.processor import Retriever
from mast3r.image_pairs import make_pairs

def extract_frames(video_path, output_dir, video_id, frame_rate=5):
    """使用ffmpeg提取视频帧"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-i', video_path,
        '-r', str(frame_rate),
        f'{output_dir}/{video_id:03d}%06d.jpg'
    ]
    subprocess.run(cmd, check=True)

def generate_adjacent_pairs(frame_dir, video_id, window_size=25):
    """生成相邻帧对"""
    frames = sorted([f for f in os.listdir(frame_dir) if f.startswith(f"{video_id:03d}")])
    pairs = []
    
    for i in range(len(frames)):
        for j in range(i+1, min(i+window_size+1, len(frames))):
            if frames[i] != frames[j]:
                pairs.append((frames[i], frames[j]))
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', required=True, nargs='+', help='输入视频列表')
    parser.add_argument('--dir', required=True, help='输出目录')
    args = parser.parse_args()

    image_dir = os.path.join(args.dir, 'images')
    all_pairs = []

    # 处理每个视频
    for video_idx, video_path in enumerate(args.videos):
        # 1. 提取视频帧
        extract_frames(video_path, image_dir, video_idx)
        
        # 2. 生成相邻帧对
        video_pairs = generate_adjacent_pairs(image_dir, video_idx)
        all_pairs.extend(video_pairs)

    # 3. 获取所有图片列表
    image_list = sorted(os.listdir(image_dir))
    
    # 4. 调用make_pairs生成场景图对
    backbone = AsymmetricMASt3R.from_pretrained("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    retriever = Retriever("checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth", backbone=backbone)
    imgs_fp = [path.join(image_dir, filename) for filename in image_list]
    with torch.no_grad():
        sim_matrix = retriever(imgs_fp)

    # Cleanup
    del retriever, backbone
    torch.cuda.empty_cache()
    scene_graph_pairs = make_pairs(image_list, scene_graph='retrieval-100-10', prefilter=None, symmetrize=True, sim_mat=sim_matrix)
    
    # 5. 合并所有配对并去重
    final_pairs = list(set(all_pairs + scene_graph_pairs))
    final_pairs = sorted(final_pairs)

    # 保存结果
    with open(os.path.join(args.dir, 'pairs.txt'), 'w') as f:
        for pair in final_pairs:
            f.write(f"{pair[0]}, {pair[1]}, 1.0\n")

if __name__ == '__main__':
    main()
