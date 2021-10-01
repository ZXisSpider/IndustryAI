import os
import mmcv
import json
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

if __name__ == '__main__':
    # Specify the path to model config and checkpoint file
    config_file = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = '../IndustryDir/epoch_50.pth'
    data_root = '../data/IndustryData/ResultSet/'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    result = []
    # test a single image and show the results
    for img in tqdm(os.listdir(data_root)):
        img_dir = data_root + img
        prediction = inference_detector(model, img_dir)
        for category in range(len(prediction)):
            if prediction[category].any():
                for flaw in prediction[category]:
                    flaw = flaw.tolist()
                    result.append({'name': img, 'category_id': category+1,
                                   'bbox': flaw[:4], 'score': flaw[-1]})
                    print(result[-1])
    with open('./result.json', 'w') as fp:
        json.dump(result, fp)



