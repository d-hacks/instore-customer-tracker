import os
import math

import cv2
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision

from .lib.config import cfg
from .lib.config import update_config
from .lib.models.pose_hrnet import get_pose_net
from .lib.utils.transforms import get_affine_transform, transform_preds
from .lib.utils.classes import COCO_KEYPOINT_INDEXES, COCO_INSTANCE_CATEGORY_NAMES


class PoseNet(object):
    def __init__(self, device, pose_model_name :str, model_weight_path :str, write_box :bool, img_size :list):
        self.device = device
        self.write_box = write_box
        self.img_size = img_size

        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        # transformation
        self.pose_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        self.box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.box_model.to(device)
        self.box_model.eval()

        print("cfg:", cfg)
        if pose_model_name == "pose_hrnet":
            self.pose_model = get_pose_net(cfg, is_train=False)
        else:
            raise NotImplementedError
        print('=> loading model from {}'.format(model_weight_path))
        self.pose_model.load_state_dict(torch.load(model_weight_path), strict=False)
        self.pose_model.to(device)
        self.pose_model.eval()

    def __call__(self, img):
        inp = img.copy()
        pred_boxes = self.get_person_detection_boxes(inp, threshold=0.9)
        # Can not find people. Move to next frame
        if not pred_boxes:
            return None, img

        if self.write_box:
            for box in pred_boxes:
                cv2.rectangle(img, box[0], box[1], color=(0, 255, 0),
                              thickness=3)  # Draw Rectangle with the coordinates

        # pose estimation : for multiple people
        centers = []
        scales = []
        for box in pred_boxes:
            center, scale = self.box_to_center_scale(box, self.img_size[0], self.img_size[1])
            centers.append(center)
            scales.append(scale)

        pose_preds = self.get_pose_estimation_prediction(inp, centers, scales)

        return pose_preds, img

    def get_person_detection_boxes(self, img, threshold=0.5):
        pil_image = Image.fromarray(img)  # Load the image
        transform = transforms.Compose([transforms.ToTensor()])  # Defing PyTorch Transform
        transformed_img = transform(pil_image)  # Apply the transform to the image
        pred = self.box_model([transformed_img.to(self.device)])  # Pass the image to the model
        # Use the first detected person
        pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                        for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                    for i in list(pred[0]['boxes'].cpu().detach().numpy())]  # Bounding boxes
        pred_scores = list(pred[0]['scores'].cpu().detach().numpy())

        person_boxes = []
        # Select box has score larger than threshold and is person
        for pred_class, pred_box, pred_score in zip(pred_classes, pred_boxes, pred_scores):
            if (pred_score > threshold) and (pred_class == 'person'):
                person_boxes.append(pred_box)

        return person_boxes

    def get_pose_estimation_prediction(self, img, centers, scales):
        rotation = 0

        # pose estimation transformation
        model_inputs = []
        for center, scale in zip(centers, scales):
            trans = get_affine_transform(center, scale, rotation, self.img_size)
            # Crop smaller image of people
            model_input = cv2.warpAffine(
                img,
                trans,
                (int(self.img_size[0]), int(self.img_size[1])),
                flags=cv2.INTER_LINEAR)

            # hwc -> 1chw
            model_input = self.pose_transform(model_input)#.unsqueeze(0)
            model_inputs.append(model_input)

        # n * 1chw -> nchw
        model_inputs = torch.stack(model_inputs)

        # compute output heatmap
        output = self.pose_model(model_inputs.to(self.device))
        coords, _ = get_final_preds(
            cfg,
            output.cpu().detach().numpy(),
            np.asarray(centers),
            np.asarray(scales))

        return coords

    def box_to_center_scale(self, box, model_image_width, model_image_height):
        """convert a box to center,scale information required for pose transformation
        Parameters
        ----------
        box : list of tuple
            list of length 2 with two tuples of floats representing
            bottom left and top right corner of a box
        model_image_width : int
        model_image_height : int

        Returns
        -------
        (numpy array, numpy array)
            Two numpy arrays, coordinates for the center of the box and the scale of the box
        """
        center = np.zeros((2), dtype=np.float32)

        bottom_left_corner = box[0]
        top_right_corner = box[1]
        box_width = top_right_corner[0]-bottom_left_corner[0]
        box_height = top_right_corner[1]-bottom_left_corner[1]
        bottom_left_x = bottom_left_corner[0]
        bottom_left_y = bottom_left_corner[1]
        center[0] = bottom_left_x + box_width * 0.5
        center[1] = bottom_left_y + box_height * 0.5

        aspect_ratio = model_image_width * 1.0 / model_image_height
        pixel_std = 200

        if box_width > aspect_ratio * box_height:
            box_height = box_width * 1.0 / aspect_ratio
        elif box_width < aspect_ratio * box_height:
            box_width = box_height * aspect_ratio
        scale = np.array(
            [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(config, batch_heatmaps, center, scale):
    coords, maxvals = get_max_preds(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    diff = np.array(
                        [
                            hm[py][px+1] - hm[py][px-1],
                            hm[py+1][px]-hm[py-1][px]
                        ]
                    )
                    coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals