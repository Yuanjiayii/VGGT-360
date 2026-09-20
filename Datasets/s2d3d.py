import os
import cv2
import numpy as np
import torch
from torchvision import transforms


def read_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = f.readlines()
        for line in lines:
            rgb_depth_list.append(line.strip().split(" ")[:2])
    return rgb_depth_list


class Stanford2D3D():
    """Stanford2D3D Dataset"""

    def __init__(
        self, 
        root_dir, 
        list_file, 
        height=512, 
        width=1024,
        device='cpu',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        ):
        """
        Args:
            root_dir (string): Directory of the Stanford2D3D Dataset.
            list_file (string): Path to the txt file contain the list of image and depth files.
            height, width: input size.
        """

        self.w = width
        self.h = height
        self.root_dir = root_dir
        self.rgb_depth_list = np.array(read_list(list_file))
        self.device = device
        self.max_depth_meters = 10.0
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = {}

        rgb_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][0])
        rgb = cv2.imread(rgb_name)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, dsize=(self.w, self.h), interpolation=cv2.INTER_CUBIC)

        depth_name = os.path.join(self.root_dir, self.rgb_depth_list[idx][1])
        gt_depth = cv2.imread(depth_name, -1)
        gt_depth = cv2.resize(gt_depth, dsize=(self.w, self.h), interpolation=cv2.INTER_NEAREST)
        if depth_name.endswith('exr') or depth_name.endswith('.npy'):
            pass
        else:
            gt_depth = gt_depth.astype(np.float32)/512
        gt_depth[gt_depth > self.max_depth_meters+1] = self.max_depth_meters + 1

        inputs["rgb"] = rgb
        inputs["gt_depth"] = torch.from_numpy(np.expand_dims(gt_depth, axis=0))
        inputs["val_mask"] = ((inputs["gt_depth"] > 0) & (inputs["gt_depth"] <= self.max_depth_meters)
                                & ~torch.isnan(inputs["gt_depth"]))
        inputs["gt_metric_depth"] = inputs["gt_depth"].clone()

        return inputs

    def __len__(self):
        return len(self.rgb_depth_list)
