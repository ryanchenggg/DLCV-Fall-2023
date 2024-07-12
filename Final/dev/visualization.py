import cv2
import torch
import json
import os

def vizualization_gt(gt_path, output_path):
    with open(gt_path, 'r') as f:
        json_data = json.load(f)    

    for video_name in json_data.keys():
        print("Processing video: ", video_name)
        video = cv2.VideoCapture(os.path.join('/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/clips', video_name+'.mp4'))

        annotation = json_data[video_name]['annotations']
        query_list = list(annotation[0]['query_sets'].values())
        for query in query_list:
            frame_list = [item for item in query['response_track']]
            frame_list.append(query['visual_crop'])
            for frame in frame_list:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame['frame_number'])
                success, image = video.read()
                if success:
                    cv2.rectangle(image, (frame['x'], frame['y']), (frame['x']+frame['width'], frame['y']+frame['height']), (0, 255, 0), 2)
                    # cv2.imshow('image', image)
                    cv2.waitKey(0)
                else:
                    print("Error reading frame: ", frame['frame_number'])
                cv2.imwrite(os.path.join(output_path, video_name+'_'+str(frame['frame_number'])+'_gt'+'.jpg'), image)
        break

def vizualization_pred(pred_path, output_path):
    with open(pred_path, 'r') as f:
        json_data = json.load(f)    

    for video_name in json_data.keys():
        print("Processing video: ", video_name)
        video = cv2.VideoCapture(os.path.join('/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/clips', video_name+'.mp4'))

        annotation_list = json_data[video_name]['predictions']
        for annotation in annotation_list:
            query_list = list(annotation['query_sets'].values())
            for predict in query_list:
                bbx_list = predict['bboxes']
                for bbx in bbx_list:
                    video.set(cv2.CAP_PROP_POS_FRAMES, bbx['fno'])
                    success, image = video.read()
                    if success:
                        cv2.rectangle(image, (bbx['x1'], bbx['y1']), (bbx['x2'], bbx['y2']), (0, 255, 0), 2)
                        # cv2.imshow('image', image)
                        cv2.waitKey(0)
                    else:
                        print("Error reading frame: ", bbx['frame_number'])
                    cv2.imwrite(os.path.join(output_path, video_name+'_'+str(bbx['fno'])+'_pred'+'.jpg'), image)

        break

if __name__ == "__main__":
    gt_path = "/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/vq_val.json"
    pred_path = "/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/VQLoC/output/ego4d_vq2d/val/validate/inference_cache_val_results.json"
    vizualization_gt(gt_path, "/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/dev/visualization")
    vizualization_pred(pred_path, "/home/leohsu-cs/DLCV2023/DLCV-Fall-2023-Final-2-boss-sdog/dev/visualization")
