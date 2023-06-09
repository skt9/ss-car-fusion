import cv2, os, argparse
from copy import deepcopy
import torch
import numpy as np
import matplotlib.cm as cm
# from src.utils.plotting import make_matching_figure
from src.loftr import LoFTR, default_cfg
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from glob import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

desired_img_dim = (1024, 512)
scaling_factor = 2.
seq_names = ['car_craig1',      #   car_butler2 is missing.
             'car_craig2',
             'car_fifth1', 
             'car_fifth2', 
             'car_morewood1', 
             'car_morewood2', 
             'car_butler1', 
             'car_penn1', 
             'car_penn2']

def make_matching_figure(
        img0, img1, 
        mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=300, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    print(f"mkpts0.shape: {mkpts0.shape} mkpts1.shape: {mkpts1.shape}")
    
    fig, axes = plt.subplots(2, 1, figsize=(2,4), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    #   
    
    # print(f"REACHED HERE")
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # print(f"kpts plotted")

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=0.1)
                                        for i in range(len(mkpts0))]
        
        # axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=2)
        # axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=2)

    # print(f"figure")

    # # put txts
    # txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    # fig.text(
    #     0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
    #     fontsize=15, va='top', ha='left', color=txt_color)
    # # plt.imshow()
    # # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig

def draw_matching_figure_cv():
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help = "Path of the dataset")
#     parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')    
#     parser.add_argument('--data_cfg_path', type=str, help='data config path')
#     parser.add_argument('--main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="weights\outdoor_ds.ckpt", help='path to the checkpoint')

    return parser.parse_args()

def get_default_configuration():
    # The default config uses dual-softmax.
    # The outdoor and indoor models share the same config.
    # You can change the default values like thr and coarse_match_type.
    _default_cfg = deepcopy(default_cfg)
    _default_cfg['coarse']['temp_bug_fix'] = True  # set to False when using the old ckpt
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load("weights/outdoor_ds.ckpt")['state_dict'])
    matcher = matcher.eval().to(device)
    return matcher

def def_value():
    return "Key not present"

def read_image(img_path, scaling_factor):
    img_raw = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print(f"img_raw.shape: {img_raw.shape}")
    img_ht, img_wt = img_raw.shape
    img_raw = cv2.resize(img_raw, desired_img_dim)
    return img_raw

def expand_to_torch4d(img_raw):
    img = torch.from_numpy(img_raw)[None][None] / 255.
    return img

def make_tiled_image(img1,img2):
    assert(img1.shape == img2.shape)
    tiled_img = np.zeros((img1.shape[0]*2, img1.shape[1]))
    tiled_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    tiled_img[img1.shape[0]:2*img1.shape[0], 0:img1.shape[1]] = img2
    return tiled_img

class CarFusionDataset:
    
    def __init__(self, args):
        dataset_path = args.dataset_path
        
        self.dataset = defaultdict(def_value)
        for seq in seq_names:
            seq_dict = defaultdict()
            seq_path = os.path.join(dataset_path, seq)
            rgb_imgs_path = os.path.join(seq_path,'images_jpg','*.jpg')
            gt_path = os.path.join(seq_path,'gt','*.txt')
            bb_path = os.path.join(seq_path,'bb','*.txt')
            rgb_imgs = glob(rgb_imgs_path)
            gt_info = glob(gt_path)
            bb_info = glob(bb_path)
            seq_dict['rgb'] = rgb_imgs
            seq_dict['gt'] = gt_info
            seq_dict['bb'] = bb_info
            self.dataset[seq] = seq_dict
    
        
    def get_sequence(self, seq: str):
        return self.dataset[seq]

    def __repr__(self):
        
        ds_str = "REPR"
        ds_str += f'------------------------------------------------------\n'
        for ky in self.dataset.keys():
            seq_dataset = self.dataset[ky]
            ds_str += f"Sequence {ky}: #rgb: {len(seq_dataset['rgb'])} #gt: {len(seq_dataset['gt'])} #bb: {len(seq_dataset['bb'])}\n"
        ds_str += f'------------------------------------------------------'
        return ds_str

def separate_camera_sequences(seq_dict):
    
    cam_dict = defaultdict()
    for rgb in seq_dict['rgb']:
        dir_part, file_part = os.path.split(rgb)
        file_base, file_ext = file_part.split('.')
        cam_id, _ = file_base.split('_', 1)
        if cam_id not in cam_dict.keys():
            cam_dict[cam_id] = []
        
        cam_dict[cam_id].append(rgb)
    
    return cam_dict

def process_sequence(seq_dataset):
    
    rgbs = seq_dataset['rgb']
    first_images = rgbs[0:len(rgbs)-1]
    second_images = rgbs[1:len(rgbs)]
    cam_dict = separate_camera_sequences(seq_dataset)
            
    seq_dir = os.path.join(args.dataset_path, seq)
    seq_results_dir = os.path.join(seq_dir, 'matched_results')
    if not os.path.exists(seq_results_dir):
        os.mkdir(seq_results_dir)

    for ky in cam_dict.keys():
        cam_img_paths = cam_dict[ky]
        first_imgs = cam_img_paths[0:len(cam_img_paths)-1]
        second_imgs = cam_img_paths[1:len(cam_img_paths)]
        
        for i, (img0_file, img1_file) in enumerate(zip(first_images, second_images)):
            
            img0_raw = read_image(first_images[i], desired_img_dim)
            img1_raw = read_image(second_images[i], desired_img_dim)
        
            img0 = expand_to_torch4d(img0_raw)
            img1 = expand_to_torch4d(img1_raw)
            batch = {'image0': img0, 'image1': img1}

            with torch.no_grad():
                matcher(batch)
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()
        
            color = cm.jet(mconf)
            text = [
                'LoFTR',
                'Matches: {}'.format(len(mkpts0)),
            ]

            fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
            
            seq_dir = os.path.join(args.dataset_path, seq)
            seq_results_dir = os.path.join(seq_dir, 'matched_results')
            if not os.path.exists(seq_results_dir):
                os.mkdir(seq_results_dir)
            
            img1_base = os.path.split(first_images[i])[1].split('.')[0]
            img2_base = os.path.split(second_images[i])[1].split('.')[0]
            
            color = cm.jet(mconf)
            text = [
                'LoFTR',
                'Matches: {}'.format(len(mkpts0)),
            ]
            
            fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text)
            matching_filename = img1_base + "_to_" + img2_base + ".jpg"  
            fig.savefig(os.path.join(seq_results_dir, matching_filename))
    

if __name__ == "__main__":
    
    matcher = get_default_configuration()
    
    args = parse_args()
    dataset = CarFusionDataset(args)
    
    for i,seq in enumerate(seq_names):
        seq_dataset = dataset.get_sequence(seq)
        process_sequence(seq_dataset)