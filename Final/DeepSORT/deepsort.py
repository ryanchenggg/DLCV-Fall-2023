from ultralytics import YOLO
import datetime
import cv2
import numpy as np
import json
from PIL import Image
from sklearn.decomposition import PCA

from tqdm.auto import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

from datasets_vq2d import VQ2DDataset, calculate_iou
from torch.utils.data import DataLoader

from deep_sort_realtime.deepsort_tracker import DeepSort, Detection
from deep_sort_realtime.embedder.mobilenetv2_bottle import MobileNetV2_bottle
from deep_sort_realtime.embedder.embedder_clip import Clip_Embedder
from deep_sort_realtime.deep_sort.nn_matching import _nn_cosine_distance, _nn_euclidean_distance
from deep_sort_realtime.deep_sort.nn_matching import NearestNeighborDistanceMetric

COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

def create_video_writer(video_cap, output_filename):

    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer

transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

val_dataset = VQ2DDataset(
                    videos_folder="../DLCV_vq2d_data/clips", 
                    annotation_path="../DLCV_vq2d_data/vq_val.json",
                    transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model settings
model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
            rpn_pre_nms_top_n_train = 2000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_train = 500,
            rpn_post_nms_top_n_test = 100,
            rpn_score_thresh = 0.7
        ).to(device)
tracker = DeepSort(embedder='clip_ViT-L/14', max_age=80)
clip_embedder = Clip_Embedder(model_name="ViT-L/14", gpu=True, bgr=False)


SIMILARITY_THRESHOLD = 70 # smaller is more strict
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
all_results = {}
bboxes_list = []

for query_lists in tqdm(val_dataloader):

    video_results = {"predictions": []}

    for query_list in query_lists:
        frame_data, video_path, video_id, bboxes, annotation_uid = query_list[0], query_list[1], query_list[2], query_list[3], query_list[4]
        # Extract query features for all queries in the frame data
        query_images = []
        frame_qas = []

        # print("video_id: ", video_id)

        for query_set in frame_data:
            query_id = query_set[0][0] # query_set[0] is the query id
            frame_qas.append(query_set[2]) # query_set[2] is the query frame number
            query_image = query_set[3].detach().cpu().numpy() # query_set[3] is the query image
            query_image = np.squeeze(query_image) * 255
            query_image = np.transpose(query_image, (1, 2, 0))
            query_image = query_image.astype(np.uint8)
            query_images.append(query_image)

        # Predict features for each query image
        query_features = [clip_embedder.predict([img]) for img in query_images]

        
        # Initialize structures for storing results
        # video_results = {"predictions": []}
        query_sets_dict = {"query_sets": {}}
        bboxes_lists = [[] for _ in range(3)]  # Initialize bbox lists for each query
        # print("len(bboxes_lists): ", len(bboxes_lists))

        # Initialize video capture and writer
        video_cap = cv2.VideoCapture(video_path[0]) #batch size = 1, so video_path[0] is the only video path
        writer = create_video_writer(video_cap, f"./deepsort_video_ry_2/{video_id[0]}_output.mp4")

        track_to_query_map = {}  # map track ID to query index
        confidence_scores_map = {query_set[0][0]: [] for query_set in frame_data}
        
        # 初始化用於存儲每個query最相似物體的資訊
        best_matches = {query_set[0][0]: {'similarity': float('inf'), 'bbox': None, 'score': None}
                    for query_set in frame_data}

        while True:
            ret, frame = video_cap.read()
            start = datetime.datetime.now()
            if not ret:
                break
            frame_number = int(video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            # print(f"Current frame number: {frame_number}")

            # Object detection logic (Faster R-CNN)
            image_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(device)
            model.eval()
            with torch.no_grad():
                prediction = model(image_tensor)
            # print(f"prediction: {len(prediction[0]['boxes'])}")
            
            frame_qas_dict = {}
            for query_set in frame_data:
                query_id = query_set[0][0]  # query_set[0] is the query id
                frame_number_qa = query_set[2]  # query_set[2] is the query frame number
                frame_qas_dict[query_id] = frame_number_qa
            
            # 更新每個查詢的最相似物體
            for query_id, match in best_matches.items():
                match['similarity'] = float('inf')  # 重置相似度

            for i, (box, score) in enumerate(zip(prediction[0]["boxes"], prediction[0]["scores"])):            
                if score < CONFIDENCE_THRESHOLD:
                    continue
                bbox = [int(coord) for coord in box.cpu().numpy()]

                # Check if the detected object matches any of the queries
                for query_idx, query_feature in enumerate(query_features):
                    query_id = frame_data[query_idx][0][0]  # 獲得 query_id
                    if frame_number >= frame_qas_dict[query_id]:
                        continue
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    crop_np = np.array(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                    crop_feature = clip_embedder.predict([crop_np])
                    similarity = _nn_euclidean_distance(query_feature, crop_feature)
                    
                    if similarity < best_matches[query_id]['similarity']:
                        best_matches[query_id]['similarity'] = similarity
                        best_matches[query_id]['bbox'] = bbox
                        best_matches[query_id]['score'] = score
                        confidence_scores_map[query_id].append(score.item())
                
            pre_filtered_tracks = []
            for query_id, match in best_matches.items():
                if match['similarity'] < SIMILARITY_THRESHOLD:
                    bbox = match['bbox']
                    score = match['score'].item()
                    bbox_format = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                    pre_filtered_tracks.append([bbox_format, score,  query_id])
                    # print(f"Best match for query {query_id} is bbox {match['bbox']} with similarity {match['similarity']}")
                    break
            
            # map query id to index
            query_id_to_index_map = {query_set[0][0]: idx for idx, query_set in enumerate(frame_data)}  

            tracks = tracker.update_tracks(pre_filtered_tracks, frame=frame)
            # print(len(tracks))
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    track_bbox = track.to_tlwh()  # DeepSort 返回的邊界框(x, y, w, h)

                    match_query_id = None
                    corresponding_score = None
                    for bbox, score, query_id in pre_filtered_tracks:
                        iou = calculate_iou(bbox, track_bbox) 
                        # print(f"iou: {iou}")
                        if iou > IOU_THRESHOLD:
                            match_query_id = query_id
                            # score取小數點後3位
                            score = round(score, 3)
                            corresponding_score = score
                            break
                    if match_query_id is not None:
                        track_to_query_map[track_id] = match_query_id
                        cv2.rectangle(frame, (int(track_bbox[0]), int(track_bbox[1])), 
                                    (int(track_bbox[0] + track_bbox[2]), int(track_bbox[1] + track_bbox[3])), GREEN, 2)
                        cv2.putText(frame, str(match_query_id), (int(track_bbox[0]) + 5, int(track_bbox[1]) - 8), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                        cv2.putText(frame, str(track_id), (int(track_bbox[0]) + 5, int(track_bbox[1]) - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                        score_text = f"score: {corresponding_score:.3f}"  # 格式化為 "score: 0.9xx"
                        cv2.putText(frame, score_text, (int(track_bbox[0]) + 5, int(track_bbox[1]) - 42),    
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
                    
                        # print(f"Track {track_id} is associated with query {match_query_id}")

                        # print("x1, y1, x2, y2: ", int(track_bbox[0]), int(track_bbox[1]), int(track_bbox[2]), int(track_bbox[3]))
                        bbox_track = {
                            "x1": int(track_bbox[0]), 
                            "y1": int(track_bbox[1]), 
                            "x2": int(track_bbox[0] + track_bbox[2]), 
                            "y2": int(track_bbox[1] + track_bbox[3]),
                            "fno": frame_number
                        }
                        if match_query_id in query_id_to_index_map:
                            bboxes_list_index = query_id_to_index_map[match_query_id]
                            bboxes_lists[bboxes_list_index].append(bbox_track)
                        else:
                            print(f"Warning: match_query_id {match_query_id} not found in query_id_to_index_map.")
                        # print(int(match_query_id)-1)
                        # bboxes_lists[int(match_query_id)-1].append(bbox_track)

            # End time to compute the FPS
            end = datetime.datetime.now()
            # print(f"FPS: {1 / (end - start).total_seconds():.2f}")
            average_confidence_scores = {}
            for query_id, scores in confidence_scores_map.items():
                avg_score = sum(scores) / len(scores) if scores else 0
                average_confidence_scores[query_id] = avg_score
                # print(f"Average confidence score for query_id {query_id}: {avg_score}")
            # Write the frame to the output
            writer.write(frame)
            if cv2.waitKey(1) == ord("q"):
                break
        
        for query_id, idx in query_id_to_index_map.items():
            if len(bboxes_lists[idx]) == 0:
                bboxes_lists[idx].append({
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0, "fno": 1
                })
            query_sets_dict["query_sets"][query_id] = {
                "bboxes": bboxes_lists[idx],
                "score": average_confidence_scores.get(query_id, 0)  # Adjust score as needed
            }

        # Finalize and save video results
        video_results["predictions"].append(query_sets_dict)
        all_results[video_id[0]] = video_results

        # Release resources
        video_cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # print(all_results)
    # input()

    # 將所有結果儲存為 JSON
    with open('pred.json', 'w') as json_file:
        json.dump(all_results, json_file, ensure_ascii=False, indent=4)