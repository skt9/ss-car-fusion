import cv2, os, glob
import numpy as np
import argparse

def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help='path to a dataset')
    parser.add_argument("--seq_name", type=str, help='sequence name')
    parser.add_argument("--frame_rate", type=int, default=5, help='Frame rate of the save video')
    parser.add_argument("--codec", type=str, default='mp4v', help='Save annotated images as video')
    args = parser.parse_args()
    return args

def convert_video(args):
    
    print(f"args.dataset_path: {args.dataset_path}")
    print(f"args.seq_name: {args.seq_name}")
    print(f"args.frame_rate: {args.frame_rate}")
    print(f"args.codec: {args.codec}")
    
    images_folder = os.path.join(args.dataset_path, args.seq_name, 'annotated_images', '*.jpg')
    print(images_folder)
    img_files = sorted(glob.glob(images_folder))
    print(len(img_files))
    
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    print(fourcc)
    
    # frameSize = (500, 500)
    if (args.codec == 'mp4v' or 'XVID' or 'DVIX' ):
        video_extension = 'mp4'
    # elif (args.codec == 'MJPG'):
    #     print(f'MJPG')
    #     video_extension = '.avi'
    # elif (args.codec == 'XVID')
    else:
        raise Exception("I don't understand video codecs.")
    
    rgb_img = cv2.imread(img_files[0])
    # print(f"rgb_img.shape: {rgb_img.shape}")
    img_width, img_height = rgb_img.shape[1], rgb_img.shape[0]
    video_file = os.path.join(args.dataset_path,args.seq_name,args.seq_name+f'_fr_{args.frame_rate}'+"."+video_extension)
    video_writer = cv2.VideoWriter(video_file, fourcc, args.frame_rate, (img_width, img_height))

    for filename in sorted(img_files):
        img = cv2.imread(filename)
        
        # print(f"img.shape: {img.shape}")
        video_writer.write(img)

    video_writer.release()


if __name__ == "__main__":
    
    args = parse_arguments()
    convert_video(args)