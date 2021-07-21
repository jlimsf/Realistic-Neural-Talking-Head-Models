import torch
from torch.utils.data import Dataset
import os
import numpy as np
import face_alignment
from PIL import Image

from .video_extraction_conversion import *

class UBCFashion(Dataset):
    def __init__(self, K, pose_root, rgb_root, transforms=None):

        self.K = K
        self.pose_root = pose_root
        self.rgb_root = rgb_root
        self.id_list = []
        self.transforms = transforms
        self.path_to_Wi = None

        for video in os.listdir(self.pose_root):
            video_dir = os.path.join(self.pose_root, video)

            this_video_list = []
            for im in os.listdir(video_dir):
                pose_im_fp = os.path.join(video_dir, im)
                rgb_im_fp = os.path.join(rgb_root, video, im)
                assert (os.path.exists(pose_im_fp))
                assert (os.path.exists(rgb_im_fp))

                this_tuple = (pose_im_fp, rgb_im_fp)
                this_video_list.append(this_tuple)
            self.id_list.append(this_video_list)
            ##not sorting frames!!


    def __len__(self):
        return len(self.id_list)

    # def black2white(self, img):
    #     black_pixels = np.where(
    #         (img[:, :, 0] == 0) &
    #         (img[:, :, 1] == 0) &
    #         (img[:, :, 2] == 0)
    #     )
    #
    #     # set those pixels to white
    #     img[black_pixels] = [255, 255, 255]
    #
    #     return img


    def __getitem__(self, idx):

        this_vid = self.id_list[idx]
        these_frames = random.sample(this_vid, self.K)

        poses = [Image.open(x[0]) for x in these_frames]
        # poses = [self.black2white(np.asarray(x)) for x in poses]
        # print (poses)
        # exit()
        rgbs = [Image.open(x[1]) for x in these_frames]


        if self.transforms:
            poses = torch.cat([self.transforms(x).unsqueeze(0) for x in poses], dim=0)
            rgbs = torch.cat([self.transforms(x).unsqueeze(0) for x in rgbs], dim= 0)


        frame_mark = torch.cat((rgbs.unsqueeze(1), poses.unsqueeze(1)), dim=1)
        g_idx = torch.randint(low = 0, high = self.K, size = (1,1))
        x = frame_mark[g_idx,0].squeeze()
        g_y = frame_mark[g_idx,1].squeeze()


        # if self.path_to_Wi is not None:
        #     try:
        #         W_i = torch.load(self.path_to_Wi+'/W_'+str(vid_idx)+'/W_'+str(vid_idx)+'.tar',
        #                     map_location='cpu')['W_i'].requires_grad_(False)
        #     except:
        #         print("\n\nerror loading: ", self.path_to_Wi+'/W_'+str(vid_idx)+'/W_'+str(vid_idx)+'.tar')
        #         W_i = torch.rand((512,1))
        # else:
        #     W_i = None

        return frame_mark, x, g_y, idx, poses, rgbs


class VidDataSet(Dataset):
    def __init__(self, K, path_to_mp4, device):
        self.K = K
        self.path_to_mp4 = path_to_mp4
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

    def __len__(self):
        vid_num = 0
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):
                for video in os.listdir(os.path.join(self.path_to_mp4, person_id, video_id)):
                    vid_num += 1
        return vid_num

    def __getitem__(self, idx):
        vid_idx = idx
        if idx<0:
            idx = self.__len__() + idx
        for person_id in os.listdir(self.path_to_mp4):
            for video_id in os.listdir(os.path.join(self.path_to_mp4, person_id)):
                for video in os.listdir(os.path.join(self.path_to_mp4, person_id, video_id)):
                    if idx != 0:
                        idx -= 1
                    else:
                        break
                if idx == 0:
                    break
            if idx == 0:
                break
        path = os.path.join(self.path_to_mp4, person_id, video_id, video)
        frame_mark = select_frames(path , self.K)
        frame_mark = generate_landmarks(frame_mark, self.face_aligner)
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3
        frame_mark = frame_mark.transpose(2,4).to(self.device)/255 #K,2,3,224,224

        g_idx = torch.randint(low = 0, high = self.K, size = (1,1))
        x = frame_mark[g_idx,0].squeeze()
        g_y = frame_mark[g_idx,1].squeeze()
        return frame_mark, x, g_y, vid_idx

class PreprocessDataset(Dataset):
    def __init__(self, K, path_to_preprocess, path_to_Wi):
        self.K = K
        self.path_to_preprocess = path_to_preprocess
        self.path_to_Wi = path_to_Wi

        self.person_id_list = os.listdir(self.path_to_preprocess)


    def __len__(self):
        vid_num = 0
        for person_id in self.person_id_list:
            # for video_id in os.listdir(os.path.join(self.path_to_preprocess, person_id)):
            #     if len(os.listdir(os.path.join(self.path_to_preprocess, person_id, video_id))) == 2*self.K:
            #         vid_num += 1
            vid_num += len(os.listdir(os.path.join(self.path_to_preprocess, person_id)))
        return vid_num-1

    def __getitem__(self, idx):
        vid_idx = idx
        if idx<0:
            idx = self.__len__() + idx
        # for person_id in self.person_id_list:
        #     for video_id in os.listdir(os.path.join(self.path_to_preprocess, person_id)):
        #         path = os.path.join(self.path_to_preprocess, person_id, video_id)
        #         if len(os.listdir(path)) == 2*self.K:
        #             if idx != 0:
        #                 idx -= 1
        #             else:
        #                 break
        #     if idx == 0:
        #         break

        path = os.path.join(self.path_to_preprocess,
                            str(idx//256),
                            str(idx)+".png")
        frame_mark = select_preprocess_frames(path)
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #K,2,224,224,3
        frame_mark = frame_mark.transpose(2,4)/255 #K,2,3,224,224

        g_idx = torch.randint(low = 0, high = self.K, size = (1,1))
        x = frame_mark[g_idx,0].squeeze()
        g_y = frame_mark[g_idx,1].squeeze()

        if self.path_to_Wi is not None:
            try:
                W_i = torch.load(self.path_to_Wi+'/W_'+str(vid_idx)+'/W_'+str(vid_idx)+'.tar',
                            map_location='cpu')['W_i'].requires_grad_(False)
            except:
                print("\n\nerror loading: ", self.path_to_Wi+'/W_'+str(vid_idx)+'/W_'+str(vid_idx)+'.tar')
                W_i = torch.rand((512,1))
        else:
            W_i = None

        return frame_mark, x, g_y, vid_idx, W_i

class FineTuningImagesDataset(Dataset):
    def __init__(self, path_to_images, device):
        self.path_to_images = path_to_images
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

    def __len__(self):
        return len(os.listdir(self.path_to_images))

    def __getitem__(self, idx):
        frame_mark_images = select_images_frames(self.path_to_images)
        random_idx = torch.randint(low = 0, high = len(frame_mark_images), size = (1,1))
        frame_mark_images = [frame_mark_images[random_idx]]
        frame_mark_images = generate_cropped_landmarks(frame_mark_images, self.face_aligner, pad=50)
        frame_mark_images = torch.from_numpy(np.array(frame_mark_images)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark_images = frame_mark_images.transpose(2,4).to(self.device) #1,2,3,256,256

        x = frame_mark_images[0,0].squeeze()/255
        g_y = frame_mark_images[0,1].squeeze()/255

        return x, g_y


class FineTuningVideoDataset(Dataset):
    def __init__(self, path_to_video, device):
        self.path_to_video = path_to_video
        self.device = device
        self.face_aligner = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device ='cuda:0')

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        path = self.path_to_video
        frame_has_face = False
        while not frame_has_face:
            try:
                frame_mark = select_frames(path , 1)
                frame_mark = generate_cropped_landmarks(frame_mark, self.face_aligner, pad=50)
                frame_has_face = True
            except:
                print('No face detected, retrying')
        frame_mark = torch.from_numpy(np.array(frame_mark)).type(dtype = torch.float) #1,2,256,256,3
        frame_mark = frame_mark.transpose(2,4).to(self.device) #1,2,3,256,256

        x = frame_mark[0,0].squeeze()/255
        g_y = frame_mark[0,1].squeeze()/255
        return x, g_y
