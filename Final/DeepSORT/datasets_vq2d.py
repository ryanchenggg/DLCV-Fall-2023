import os
import torch
from torch.utils.data import Dataset
import json
import cv2
from PIL import Image

class VQ2DDataset(Dataset):
    def __init__(self, videos_folder, annotation_path, transform=None):
        self.videos_folder = videos_folder
        self.transform = transform

        # 加載 JSON 數據
        with open(annotation_path, 'r') as file:
            self.data = json.load(file)

        self.video = None
        self.video_ids = list(self.data.keys())

    def __len__(self):
        # 可以根據需要調整，例如返回查詢總數
        return len(self.video_ids)

    def __getitem__(self, idx):

        video_id = self.video_ids[idx]
        video_file_path = os.path.join(self.videos_folder, f"{video_id}.mp4")

        # 根據當前的 video_id 打開相應的影片
        if self.video is not None:
            self.video.release()  # 釋放先前打開的影片資源
        self.video = cv2.VideoCapture(video_file_path)

        query_list = []

        annotations = self.data[video_id]['annotations']
        for i in range(len(annotations)):
            annotation = annotations[i] # only one annotations(include all) in each video
            query_sets = annotation['query_sets']
            annotation_uid = annotation["annotation_uid"]

            frames_data = []
            bboxes = []
            for query_id, query_info in query_sets.items():
                
                frame_number_qa = query_info['query_frame']
                object_title = query_info['object_title']
            
                # 處理 visual_crop
                visual_crop_info = query_info['visual_crop']
                crop_frame_num = visual_crop_info['frame_number']

                self.video.set(cv2.CAP_PROP_POS_FRAMES, crop_frame_num)
                ret, crop_frame = self.video.read()
                if ret:
                    crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
                    crop_frame = Image.fromarray(crop_frame)

                    x, y, w, h = visual_crop_info['x'], visual_crop_info['y'], visual_crop_info['width'], visual_crop_info['height']
                    bbox = (x, y, x + w, y + h)
                    bboxes.append(bbox)
                    crop_image = crop_frame.crop((x, y, x + w, y + h))

                    if self.transform:
                        # query_frame = self.transform(query_frame)
                        crop_image = self.transform(crop_image)

                    frames_data.append((query_id, object_title, frame_number_qa, crop_image))

            query_list.append([frames_data, video_file_path, video_id, bboxes, annotation_uid])

        # return frames_data, video_file_path, video_id, bboxes, annotation_uid
        return query_list

    def close(self):
        if self.video is not None:
            self.video.release()

def calculate_iou(box1, box2):
    """計算兩個邊界框之間的 IoU。"""
    # 計算交集的座標
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[0] + box1[2], box2[0] + box2[2])
    y_bottom = min(box1[1] + box1[3], box2[1] + box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # 計算交集區域
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # 計算聯合區域
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area

    # 計算 IoU
    iou = intersection_area / union_area
    return iou

def extract_continuous_bboxes(bboxes, continuity_threshold=5):
    """
    提取连续的边界框序列。
    continuity_threshold: 允许的最大间隔帧数。
    """
    if not bboxes:
        return []

    sorted_bboxes = sorted(bboxes, key=lambda x: x['fno'])
    continuous_bboxes = []
    current_sequence = [sorted_bboxes[0]]

    for i in range(1, len(sorted_bboxes)):
        if sorted_bboxes[i]['fno'] - sorted_bboxes[i-1]['fno'] <= continuity_threshold:
            current_sequence.append(sorted_bboxes[i])
        else:
            if len(current_sequence) > 1:
                continuous_bboxes.extend(current_sequence)
            current_sequence = [sorted_bboxes[i]]
    
    if len(current_sequence) > 1:
        continuous_bboxes.extend(current_sequence)
    
    return continuous_bboxes

def is_bbox_in_continuous_time(last_frame, current_frame, threshold=5):
    """
    判断边界框是否在连续时间帧内。
    last_frame: 上一次出现的帧编号。
    current_frame: 当前帧编号。
    threshold: 允许的最大间隔帧数。
    """
    return current_frame - last_frame <= threshold