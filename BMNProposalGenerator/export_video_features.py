# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pandas as pd
import json
import torch.utils.data as data
import torch
#from .utils import ioa_with_anchors, iou_with_anchors
import lmdb
from os.path import join
import numpy, scipy.interpolate
import os.path

def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--path_to_dataset',
        type=str
    )

    parser.add_argument(
        '--rgb_lmdb',
        type=str
    )

    parser.add_argument(
        '--flow_lmdb',
        type=str
    )

    parser.add_argument(
        '--path_to_video_features',
        type=str,
        default='video_features'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='train'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=30
    )

    parser.add_argument(
        '--sigma',
        type=int,
        default=15
    )

    parser.add_argument(
        '--fname_template',
        type=str,
        default='frame_{:010d}.jpg'
    )

    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.001
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4
    )

    parser.add_argument(
        '--train_epochs',
        type=int,
        default=9
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--step_size',
        type=int,
        default=7
    )

    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--observation_window',
        type=int,
        default=400
    )

    parser.add_argument(
        '--max_duration',
        type=int,
        default=400
    )

    parser.add_argument(
        '--num_sample',
        type=int,
        default=4
    )

    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3
    )

    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5
    )

    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048
    )

    # Post processing
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=32
    )

    parser.add_argument(
        '--soft_nms_alpha',
        type=float,
        default=0.4
    )

    parser.add_argument(
        '--soft_nms_low_thres',
        type=float,
        default=0.5
    )

    parser.add_argument(
        '--post_processing_topk',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--max_proposals',
        type=int,
        default=1000
    )

    parser.add_argument(
        '--ppm',
        type = int,
        default = 200
    )

    parser.add_argument(
        '--soft_nms_high_thres',
        type=float,
        default=0.9
    )

    parser.add_argument(
        '--result_file',
        type=str,
        default="/result_proposal.pkl"
    )

    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="/evaluation_result.jpg"
    )

    parser.add_argument(
        '--inference_set',
        type=str,
        default="validation"
    )

    args = parser.parse_args()

    return args



def resizeFeature(inputData,newSize):
    # inputX: (temporal_length,feature_dimension) #
    originalSize=len(inputData)
    #print originalSize
    if originalSize==1:
        inputData=np.reshape(inputData,[-1])
        return np.stack([inputData]*newSize)
    x=numpy.array(range(originalSize))
    f=scipy.interpolate.interp1d(x,inputData,axis=0)
    x_new=[i*float(originalSize-1)/(newSize-1) for i in range(newSize)]
    y_new=f(x_new)
    return y_new

def ioa_with_anchors(anchors_min, anchors_max, box_min, box_max):
    # calculate the overlap proportion between the anchor and all bbox for supervise signal,
    # the length of the anchor is 0.01
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    scores = np.divide(inter_len, len_anchors)
    return scores


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    # print inter_len,union_len
    jaccard = np.divide(inter_len, union_len)
    return jaccard

def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


class VideoDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.observation_window = opt["observation_window"]  # 100
        self.temporal_gap = 1. / self.observation_window
        self.max_duration = opt["max_duration"]
        self.subset = subset
        self.mode = opt["mode"]
        if opt['rgb_lmdb']:
            self.env_rgb = lmdb.open(opt["rgb_lmdb"], readonly=True, lock=False)
        else:
            self.env_rgb = None
        if opt['flow_lmdb']:
            self.env_flow = lmdb.open(opt["flow_lmdb"], readonly=True, lock=False)
        else:
            self.env_flow = None
        self._get_match_map()
        if subset=='train':
            sset='training'
        elif subset=='validation':
            sset='validation'
        elif subset == 'test':
            sset = 'test_timestamps'
        else:
            raise Exception("subset must be in ['train','validation','test']")
        self.annotations = pd.read_csv(join(opt["path_to_dataset"], sset + '.csv'), names=['id', 'video', 'start', 'stop', 'verb', 'noun', 'action'], index_col='id')
        if isinstance(self.annotations.iloc[0]['start'], str):
            self.annotations = pd.read_csv(join(opt["path_to_dataset"], sset + '.csv'), index_col='narration_id')
        self.video_list = [v.strip() for v in self.annotations['video'].unique()]
        self.lengths = pd.read_csv(join(opt["path_to_dataset"], 'video_lengths.csv'))
        self.length_dict = self.lengths = self.lengths.set_index('video').to_dict()['frames']
        self.fname_template=opt["fname_template"]
        self.fps=opt["fps"]
        self.opt=opt

        # Read duration.
        self.durations = pd.read_csv('/afs/cs.pitt.edu/usr0/key36/work2/EK100/epic-kitchens-100-annotations/EPIC_100_video_info.csv')
        self.duration_dict = self.durations.set_index('video_id').to_dict()['duration']

    def _get_feats(self, fname):
        with self.env_rgb.begin() as e:
            dd = e.get(fname.encode())
        if dd is not None:
            rgb = np.frombuffer(dd, dtype='float32').reshape(-1, 1)
        else:
            print('rgb feat is none')
            print(fname)
            rgb = torch.zeros((1024,1))
        with self.env_flow.begin() as e:
            dd = e.get(fname.encode())
        if dd is not None:
            flow = np.frombuffer(dd, dtype='float32').reshape(-1,1)
        else:
            print('flow feat is none')
            flow = torch.zeros(*rgb.shape)
        res = np.concatenate([rgb, flow])
        return res

    def _load_file(self, index):
        video_name = self.video_list[index]
        if not os.path.isfile(f"{self.opt['path_to_video_features']}/{video_name}.npy"):
            length = self.length_dict[video_name]

            feats = []
            ff = np.arange(length-1)+1
            for f in ff:
                fname = video_name + '_' + self.fname_template.format(f)
                feats.append(self._get_feats(fname))
            video_data = np.hstack(feats)

            duration = int(self.duration_dict[video_name] / 1)
            video_data = resizeFeature(video_data.T, duration).T

            np.save(f"{self.opt['path_to_video_features']}/{video_name}.npy", video_data)
        else:
            video_data = np.load(f"{self.opt['path_to_video_features']}/{video_name}.npy")
        return video_data

    def __getitem__(self, index):
        video_data = self._load_file(index)
        if self.mode == "train":
            match_score_start, match_score_end, confidence_score = self._get_train_label(index, self.anchor_xmin,
                                                                                         self.anchor_xmax)
            return video_data,confidence_score, match_score_start, match_score_end
        else:
            return index, video_data

    def _get_match_map(self):
        match_map = []
        for idx in range(self.max_duration):
            tmp_match_window = []
            xmin = self.temporal_gap * idx
            for jdx in range(1, self.observation_window + 1):
                xmax = xmin + self.temporal_gap * jdx
                tmp_match_window.append([xmin, xmax])
            match_map.append(tmp_match_window)
        match_map = np.array(match_map)  # 100x100x2
        match_map = np.transpose(match_map, [1, 0, 2])  # [0,1] [1,2] [2,3].....[99,100]
        match_map = np.reshape(match_map, [-1, 2])  # [0,2] [1,3] [2,4].....[99,101]   # duration x start
        self.match_map = match_map  # duration is same in row, start is same in col
        self.anchor_xmin = [self.temporal_gap * (i-0.5) for i in range(self.observation_window)]
        self.anchor_xmax = [self.temporal_gap * (i+0.5) for i in range(1, self.observation_window + 1)]

    def _get_train_label(self, index, anchor_xmin, anchor_xmax):
        video_name = self.video_list[index]
        video_frame = self.observation_window
        video_second = self.length_dict[video_name]/self.opt['fps']
        feature_frame = video_frame
        #video_info = self.video_dict[video_name]
        #video_frame = video_info['duration_frame']
        #video_second = video_info['duration_second']
        #feature_frame = video_info['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second  # there are some frames not used
        video_labels = self.annotations[self.annotations['video']==video_name][['start', 'stop']].values/self.opt['fps']
        #video_labels = video_info['annotations']  # the measurement is second, not frame

        ##############################################################################################
        # change the measurement from second to percentage
        gt_bbox = []
        gt_iou_map = []
        for j in range(len(video_labels)):
            seg = video_labels[j]
            tmp_start = max(min(1, seg[0] / corrected_second), 0)
            tmp_end = max(min(1, seg[1] / corrected_second), 0)
            gt_bbox.append([tmp_start, tmp_end])
            tmp_gt_iou_map = iou_with_anchors(
                self.match_map[:, 0], self.match_map[:, 1], tmp_start, tmp_end)
            tmp_gt_iou_map = np.reshape(tmp_gt_iou_map,
                                        [self.max_duration, self.observation_window])
            gt_iou_map.append(tmp_gt_iou_map)
        gt_iou_map = np.array(gt_iou_map)
        gt_iou_map = np.max(gt_iou_map, axis=0)
        gt_iou_map = torch.Tensor(gt_iou_map)
        ##############################################################################################

        ####################################################################################################
        # generate R_s and R_e
        gt_bbox = np.array(gt_bbox)
        gt_xmins = gt_bbox[:, 0]
        gt_xmaxs = gt_bbox[:, 1]
        gt_lens = gt_xmaxs - gt_xmins
        gt_len_small = 3 * self.temporal_gap  # np.maximum(self.temporal_gap, self.boundary_ratio * gt_lens)
        gt_start_bboxs = np.stack((gt_xmins - gt_len_small / 2, gt_xmins + gt_len_small / 2), axis=1)
        gt_end_bboxs = np.stack((gt_xmaxs - gt_len_small / 2, gt_xmaxs + gt_len_small / 2), axis=1)
        #####################################################################################################

        ##########################################################################################################
        # calculate the ioa for all timestamp
        match_score_start = []
        for jdx in range(len(anchor_xmin)):
            match_score_start.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_start_bboxs[:, 0], gt_start_bboxs[:, 1])))
        match_score_end = []
        for jdx in range(len(anchor_xmin)):
            match_score_end.append(np.max(
                ioa_with_anchors(anchor_xmin[jdx], anchor_xmax[jdx], gt_end_bboxs[:, 0], gt_end_bboxs[:, 1])))
        match_score_start = torch.Tensor(match_score_start)
        match_score_end = torch.Tensor(match_score_end)
        ############################################################################################################

        return match_score_start, match_score_end, gt_iou_map

    def __len__(self):
        return len(self.video_list)


if __name__ == '__main__':
    opt = parse_opt()
    opt = vars(opt)

    if not os.path.exists(opt["path_to_video_features"]):
        os.makedirs(opt["path_to_video_features"])

    for subset in ['train', 'validation', 'test']:
      data_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset=subset), 
          batch_size=opt["batch_size"], shuffle=False,
          num_workers=8)
      for a,b,c,d in data_loader:
          print(a.shape,b.shape,c.shape,d.shape)
