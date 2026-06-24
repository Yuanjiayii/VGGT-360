import argparse
import os
import torch.utils
import torch.utils.data
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
np.bool = np.bool_
np.float = np.float32
from Datasets import Stanford2D3D
from utils.metric import Affine_Inv_Evaluator
from vggt_visfeat.models.vggt import VGGT
from utils.gen_views import view_generation
from utils.ERP_utils import ERP2Persp, depth_set_to_equirect_attention
from utils.att_utils import SA_confidence
from PIL import Image
from vggt_visfeat.utils.load_fn2 import load_and_preprocess_images


def val(
    args: argparse.ArgumentParser.parse_args,
    model: nn.Module,
    dataloader: DataLoader,
    evaluator: Affine_Inv_Evaluator=None,
    mode='Valid'):

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader)
        pbar.set_description("{}".format(mode))
        for batch_idx, inputs in enumerate(pbar):
            """ERP2Persp"""
            pano_img = inputs["rgb"][0].numpy()
            pano_H, pano_W, _ = pano_img.shape
            view_params = view_generation(pano_img, args.FOV)

            persp_imgs = []
            SA_masks = []
            valid_masks = []
            for i, (yaw, pitch, fov) in enumerate(view_params):
                persp_img = ERP2Persp(pano_img, FOV=fov, THETA=yaw, PHI=pitch, height=512, width=512, mode='bilinear')
                SA_mask, valid_mask = SA_confidence(persp_img)
                persp_imgs.append(persp_img)
                SA_masks.append(SA_mask)
                valid_masks.append(valid_mask)
            SA_masks = torch.from_numpy(np.array(SA_masks))
            valid_masks = torch.from_numpy(np.array(valid_masks))

            """VGGT process"""
            persp_imgs_list = [Image.fromarray(np.clip(face, 0, 255).astype(np.uint8)) for face in persp_imgs]
            persp_imgs_tensor = load_and_preprocess_images(persp_imgs_list).to(args.device)

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions, attention_maps = model(
                        images=persp_imgs_tensor,
                        persp_masks=SA_masks,
                        rgb_masks=valid_masks,
                    )

            """Persp2ERP"""
            world_points_array = predictions["world_points"][0].cpu().numpy()
            radial_distances = np.linalg.norm(world_points_array, axis=-1).astype(np.float32)

            erp_depth = depth_set_to_equirect_attention(
                depths=radial_distances,
                view_params=view_params,
                attention_maps=attention_maps,
                H=pano_H, W=pano_W,
                interp='linear',
            )

            erp_depth[erp_depth == np.inf] = 0
            if inputs["val_mask"].sum() > 0:
                evaluator.compute_affine_inv_eval_metrics(
                    gt_depth=inputs["gt_metric_depth"].detach(),
                    pred_depth=torch.from_numpy(erp_depth[None,None,:]),
                    mask=inputs["val_mask"].detach())

        evaluator.print()



def main():
    parser = ArgumentParser(description="VGGT360 parameters")
    parser.add_argument('--db_nm', type=str, default="S2D3D")
    parser.add_argument('--FOV', type=int, default=110)
    parser.add_argument('--model_path', type=str, default='facebook/VGGT-1B')
    parser.add_argument('--zero_shot_root', type=str, default='Stanford2D3D/stanford2d3d_rgbd')   
    parser.add_argument('--zero_shot_txt', type=str, default='/splits2d3d/stanford2d3d_test.txt')   
    parser.add_argument('--pano_h', type=str,default=512)
    parser.add_argument('--pano_w', type=str, default=1024)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    args, _ = parser.parse_known_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_dir, exist_ok=True)
    model = VGGT.from_pretrained(args.model_path)
    model = model.to(args.device)

    zero_shot_dataset = Stanford2D3D(
        root_dir=args.zero_shot_root,
        list_file=args.zero_shot_txt,
        height=args.pano_h,
        width=args.pano_w,
        device=args.device)
    zeroshot_loader = DataLoader(zero_shot_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    zeroshot_evaluator = Affine_Inv_Evaluator(median_align=True, crop=68)

    if zeroshot_loader is not None:
        zeroshot_evaluator.reset_eval_metrics()
        val(args, model, zeroshot_loader, zeroshot_evaluator, mode='Zeroshot')


if __name__ == '__main__':
    main()
