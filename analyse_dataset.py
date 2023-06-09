import cv2
import numpy as np
import torch
import glob
import sys,os
import argparse
from typing import List
#   
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
# from colorspacious import cspace_converter

#   CONFIG VARIABLES
img_dim = 512
threshold = 20
scaling_factor = 2.

correspondence  = [0,1,2,3,4,5,6,7,14,10,11,12,13]        
gt_indices  = [1,2,3,4,5,6,7,8,9,10,11,12,13]        
correspondence  = [1,0,3,2,5,4,7,6,14,11,10,13,12]    

#   Sequence names
seq_names =['car_craig1', 
            'car_craig2', 
            'car_fifth1', 
            'car_fifth2', 
            'car_penn1', 
            'car_penn2', 
            'car_morewood1', 
            'car_morewood2', 
            'car_butler1']

class ObjKeypoints:
    
    def __init__(self, kpts: np.array):
        self.kpts = kpts

class Keypoint:
    
    def __init__(self, x: float, y: float, id1: int, id2: int, id3: int):
        self.x = x
        self.y = y
        self.kpt_id = id1
        self.obj_id = id2
        self.id3 = id3
        
    def __repr__(self):
        kpt_str = f''
        kpt_str += f'x: {self.x} y: {self.y} '
        kpt_str += f'obj_id: {self.obj_id} kpt_id: {self.kpt_id} id3: {self.id3}'
        return kpt_str

def get_keypoints_ground_truth(file_name):
    
    EPS = 1e-6
    with open(file_name, 'r') as content_file:
        lines = content_file.readlines()
    keypoints = []
    for line in lines:
        keypoint = line.rsplit('\n')[0]
        # print(keypoint)
        kpt_data = keypoint.split(',')
        # print(kpt_data)
        x, y = float(kpt_data[0]), float(kpt_data[1])
        id1, id2, id3 = int(kpt_data[2]), int(kpt_data[3]), int(kpt_data[4])
        kpt = Keypoint(x, y, id1, id2, id3)
        # print(kpt)
        keypoints.append(kpt)
    
        max_kpts = max(kpt.kpt_id for kpt in keypoints)
        max_obj = max(kpt.obj_id for kpt in keypoints)
        max_id3 = max(kpt.id3 for kpt in keypoints)
        
    obj_keypoints = [np.zeros((16,3)) for obj_id in range(max_obj+1)]        

    for kpt in keypoints:
        obj_id = kpt.obj_id
        kpt_id = kpt.kpt_id
        x, y = np.round(kpt.x), np.round(kpt.y)
        if (int(x)==0 & int(y)==0):
            continue
        obj_keypoints[obj_id][kpt_id-1,0] = kpt.x
        obj_keypoints[obj_id][kpt_id-1,1] = kpt.y
        obj_keypoints[obj_id][kpt_id-1,2] = kpt_id-1

    
    obj_keypoints = [obj_kpts[~np.all(np.abs(obj_keypoints[0]) < EPS, axis=1)] for obj_kpts in obj_keypoints]
    # obj_keypoints[0] = obj_keypoints[0][~np.all(np.round(obj_keypoints).astype(int) == 0, axis=1)]
    return obj_keypoints

def scale_keypoints(kpts_list: List[np.array], scaling_factor: np.float32):
    for obj_kpts in kpts_list:
        xy_coods = obj_kpts[:,0:2]/scaling_factor
        obj_kpts[:,0:2]= xy_coods
    
    return kpts_list

def filter_keypoints(kpts_list):
    filtered_kpts_list = []
    for obj_kpts in kpts_list:
        summed_mask = np.sum(obj_kpts < 0,axis=1)
        filtered_obj_kpts = obj_kpts[summed_mask==0,:]
        filtered_kpts_list.append(filtered_obj_kpts)
    return filtered_kpts_list

def get_colormap():

    colors =  plt.cm.Vega20c( (4./3*np.arange(20*3/4)).astype(int) )
    plt.scatter(np.arange(15),np.ones(15), c=colors, s=180)
    plt.show()


def draw_keypoints_on_car(img: np.array, kpts_list: List[np.array]):
    '''
        Draw the keypoints on cars.

    '''
    for obj_kpts in kpts_list:
        for i in range(obj_kpts.shape[0]):
            kpt = obj_kpts[i,:]
            circle_radius = 2
            circle_thickness = 2
            img = cv2.circle(img, tuple([int(kpt[0]), int(kpt[1])]), circle_radius, (255,255,255), circle_thickness)
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX     # font
            cpt = (int(kpt[0]), int(kpt[1]))    # org   
            fontScale = 0.7                     # fontScale
            color = (255, 0, 255)               # Blue color in BGR
            thickness = 2                       # Line thickness of 2 px
            
            img = cv2.putText(img, str(int(kpt[2])), cpt, font, 
                   fontScale, color, thickness, cv2.LINE_AA)

    return img

def file_parts(file_name):
    file_base, file_ext = file_name.split('.')
    seq_id, img_id = file_base.split('_')
    # print(f"seq_id: {seq_id} img_id: {img_id} file_ext: {file_ext}")
    return seq_id, img_id, file_ext

def partition_into_subsequences(img_files):

    scene_decomposition = {}
    for img_file in img_files:
        _, file_name = os.path.split(img_file)
        seq_id, img_id, file_ext = file_parts(file_name)
        if (seq_id not in scene_decomposition.keys()):
            scene_decomposition[seq_id] = []
        scene_decomposition[seq_id].append(img_id)
    return scene_decomposition

def process_sequence(args, seq_name):

    dataset_path = args.dataset_path
    dir_path = os.path.join(dataset_path,seq_name)

    if os.path.exists(dir_path):
        bb_path = os.path.join(dir_path,'bb')
        imgs_path = os.path.join(dir_path,'images_jpg')
        gt_path = os.path.join(dir_path,'gt')
    else:
        raise Exception("Sequence folder does not exist.")

    img_files = glob.glob(os.path.join(imgs_path,'*.jpg'))    
    gt_labels = sorted(glob.glob(os.path.join(gt_path,'*.txt')))
    bbs = glob.glob(os.path.join(bb_path,'*.txt'))

    annotated_images_path = os.path.join(dir_path,'annotated_images')
    if not os.path.exists(annotated_images_path):
        os.mkdir(annotated_images_path)
    else:
        # raise Exception("Annotated images path already exists.")
        print("Annotated images path already exists. Continuing...")

    videos_main_path = os.path.join(args.dataset_path,'sequence_videos')
    videos_seq_path = os.path.join(videos_main_path,seq_name)
    
    scene_decomposition  = partition_into_subsequences(gt_labels)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_extension = 'mp4'
    
    for camera in scene_decomposition.keys():
        img_list = scene_decomposition[camera]
        video_file = os.path.join(args.dataset_path,'sequence_videos',seq_name+f'-c{camera}-' +f'fr_{args.frame_rate}'+"."+video_extension)
        img_name = camera + '_' + img_list[0] + ".jpg"
        img_file_fullpath = os.path.join(imgs_path, img_name)
        rgb_img = cv2.imread(img_file_fullpath)
        rgb_img = cv2.resize(rgb_img, (round(rgb_img.shape[1]/scaling_factor), round(rgb_img.shape[0]/scaling_factor)))        
        img_width, img_height = rgb_img.shape[1], rgb_img.shape[0]
        video_writer = cv2.VideoWriter(video_file, fourcc, args.frame_rate, (img_width, img_height))
        for img_base in img_list:
            #   Read the image and kpts
            img_name = camera + '_' + img_base + ".jpg"
            gt_label = camera + '_' + img_base + ".txt"
            img_file_fullpath = os.path.join(imgs_path, img_name)
            gt_file_fullpath = os.path.join(gt_path, gt_label)
            kpts_list = get_keypoints_ground_truth(gt_file_fullpath)
            rgb_img = cv2.imread(img_file_fullpath)
            
            #   Resize the image, rescale the keypoints
            rgb_img = cv2.resize(rgb_img, (round(rgb_img.shape[1]/scaling_factor), round(rgb_img.shape[0]/scaling_factor)))        
            img_width, img_height = rgb_img.shape[1], rgb_img.shape[0]
            kpts_list = scale_keypoints(kpts_list, scaling_factor)
            kpts_list = filter_keypoints(kpts_list)
            
            #   Draw keypoints on the car
            rgb_img = draw_keypoints_on_car(rgb_img, kpts_list)
            #   Write the annotated image to disk
            annotated_image_file = os.path.join(dataset_path,  'annotated_images', img_name)
            cv2.imwrite(annotated_image_file, rgb_img)
            #   Write to the video writer   
            video_writer.write(rgb_img)
            
        #   Release the video
        video_writer.release()
        

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='path to a dataset')
    # parser.add_argument("--seq_name", type=str, help='Sequence of the dataset')
    parser.add_argument("--save-images", type=bool, default=True, help='Save annotated images from the dataset')
    parser.add_argument("--save-videos", type=bool, default=True, help='Save annotated images as video')
    parser.add_argument("--frame_rate", type=int, default=1, help='Frame rate of the save video')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    args = parse_arguments()
    for i in range(1,len(seq_names)):
        process_sequence(args, seq_names[i])