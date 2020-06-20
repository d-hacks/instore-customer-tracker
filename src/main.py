import copy
import os

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from typing import NewType, Tuple
import yaml

from modules.posenet import PoseNet
from modules.utils import Config

VideoRead = NewType('VideoRead', cv2.VideoCapture)
VideoWrite = NewType('VideoWrite', cv2.VideoWriter)


def main(cfg):
    cfg = Config(cfg)

    print("Loading '{}'".format(cfg.input))

    # Load the video
    cap, origin_fps, frames, width, height = load_video(cfg.input)
    writer = video_writer(cfg.output_path, origin_fps,
                            width, height)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model
    posenet = PoseNet(device, cfg.pose_model_name, cfg.pose_model_weight, cfg.write_box, cfg.img_size)

    count = 1
    while(cap.isOpened()):
        print('')
        ret, frame = cap.read()
        if not ret:
            break

        print('-----------------------------------------------------')
        print('[INFO] Count: {}/{}'.format(count, frames))

        # posenet preediction
        poses, out = posenet(np.array(F.to_pil_image(frame)))
        if poses is not None:
            print(poses.shape, out.shape)

        # extract or predict foot position

        # map to 2d space

        # kalman filter

        # id assign

        # append to trajectoeries

        writer.write(out)
        count += 1

    # Stop video process
    cap.release()
    writer.release()


def load_video(path: str) -> Tuple[VideoRead, int, int, int, int]:
    print('[INFO] Loading "{}" ...'.format(path))
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print('[INFO] total frame: {}, fps: {}, width: {}, height: {}'.format(frames, fps, W, H))
    return cap, fps, frames, W, H
    

def video_writer(path: str, fps: int, width: int, height: int,
                 resize_factor: float =None) -> VideoWrite:
    print("[INFO] Save output video in", path)
    if resize_factor is not None:
        width = int(width * resize_factor)
        height = int(height * resize_factor)
        width -= width % 4
        height -= height % 4
    # height *= 2
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    writer = cv2.VideoWriter(path, fourcc, fourcc, (width, height))
    return writer


if __name__ == '__main__':
    with open('src/config.yaml') as f:
        cfg = yaml.safe_load(f)
    main(cfg)
