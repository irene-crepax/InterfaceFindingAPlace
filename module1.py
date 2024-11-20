import imutils
import pytesseract
import json
import os
import cv2
import torch
import pandas as pd
import streamlit as st

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"
class_map = {0: "caption", 1: "text", 2: "figure", 3: "title"}


#@st.cache_data
def create_empty_annotations(directory):
    images = list()
    i = 0
    for image in os.listdir(directory):
        try:
            img = cv2.imread(os.path.join(directory, image))
            h = img.shape[0]
            w = img.shape[1]
        except Exception as _:
            continue
        images.append({"width": w, "height": h,
                       "id": i, "file_name": image})
        i += 1

    json_object = json.dumps(images, indent=2)

    with open('_'.join([os.path.split(directory)[1], "annotations.json"]), "w") as outfile:
        outfile.write(json_object)


def get_prima_dicts(img_dir):
    json_file = '_'.join([os.path.split(img_dir)[1], "annotations.json"])
    f = open(json_file)
    images = json.load(f)
    dataset_dicts = list()
    for item in images:
        record = dict()
        record["file_name"] = os.path.join(img_dir, item['file_name'])
        record["image_id"] = item['id']
        record["height"] = item['height']
        record["width"] = item['width']
        dataset_dicts.append(record)
    return dataset_dicts


def write_to_dict(outputs, filename, labels_path):
    boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist()
    classes = outputs["instances"].pred_classes.cpu().numpy().tolist()
    classes = [class_map[i] for i in classes]
    new_boxes = list()
    for box in boxes:
        new_boxes.append([[box[0], box[1]], [box[2], box[1]],
                          [box[2], box[3]], [box[0], box[3]]])
    shapes = [{"label": classes[i], "points": new_boxes[i]}
              for i in range(len(classes))]
    annos = {'filename': filename, 'annotations': shapes}
    jsonobject = json.dumps(annos)
    with open(os.path.join(labels_path,'.'.join(os.path.split(filename)[1].split('.')[:-1]) + '.json'), 'w') as f:
        f.write(jsonobject)


#@st.cache_data
def predict(test_path, labels_path, name, model_path, output_path):
    dataname = f"{name}_predict"
    if dataname not in DatasetCatalog.list():
        DatasetCatalog.register(dataname, lambda: get_prima_dicts(test_path))
        MetadataCatalog.get(dataname).set(thing_classes=list(class_map.values()))

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TEST = (name + "_predict",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 10000
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # path to the model we just trained
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    balloon_metadata = MetadataCatalog.get(dataname)
    dataset_dicts = DatasetCatalog.get(dataname)
    existing_labels = os.listdir(labels_path)
    for dataset_dict in dataset_dicts:
        name = os.path.basename(dataset_dict["file_name"])
        if name.split('.')[0] + '.json' in existing_labels:
            continue

        im = cv2.imread(dataset_dict["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5
                       )
        write_to_dict(outputs, name, labels_path)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out = v.draw_dataset_dict(dataset_dict)
        cv2.imwrite(os.path.join(output_path, name), out.get_image()[:, :, ::-1])


#@st.cache_data
def generate_caption(labels_path, output, images_path, target):
    all_captions = dict()
    all_labels = os.listdir(labels_path)
    ext = os.listdir(images_path)[0].split('.')[-1]
    for label in all_labels:
        captions_dict = dict()
        img_path = label.replace('json', ext)
        img = cv2.imread(os.path.join(images_path, img_path))
        with open(os.path.join(labels_path, label)) as json_file:
            data = json.load(json_file)
            index = 0
            for item in data['annotations']:
                if item['label'] == target:
                    pts_list = [[int(pt) for pt in coords]
                                for coords in item['points']]
                    x = [item[0] for item in pts_list]
                    y = [item[1] for item in pts_list]
                    x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                    image = img[y1:y2, x1:x2]
                    confs = dict()
                    for angle in [0, 90, 180, 270]:
                        im = imutils.rotate_bound(image, angle)
                        text = pytesseract.image_to_data(
                            im, config='--oem 3 --psm 6', output_type='data.frame')
                        text = text[text.conf != -1]
                        conf = text.groupby(['block_num'])['conf'].mean()
                        confs[angle] = conf.values
                        if confs[angle].size == 0:
                            confs[angle] = 0
                    confs_values = list(confs.values())
                    conf = max(confs_values)
                    angle = [i for i in confs if confs[i] == conf]
                    im = imutils.rotate_bound(image, angle=angle[0])
                    ocr_result = pytesseract.image_to_string(im)
                    captions_dict[index] = [ocr_result, angle[0]]
                    index += 1

        all_captions[img_path] = captions_dict

    caption_dict = dict()
    for i, (page, features) in enumerate(all_captions.items()):
        caption_dict[i] = dict()
        for data_index in features:
            caption_dict[i]["Page"] = page
            caption_dict[i]["text_{}".format(data_index)] = features[data_index][0]
            caption_dict[i]["rotation_{}".format(data_index)] = features[data_index][1]
    captions_df = pd.DataFrame.from_dict(caption_dict, orient="index")
    captions_df.to_csv(output, sep=",", index=False)


#@st.cache_data
def parsing_caption(captions_path, labels_path, output, ocr_target):
    captions_df = pd.read_csv(captions_path)
    all_labels = list()
    for label_path in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_path), 'r') as f:
            label = json.load(f)
            filename = label['filename']
            annos = label['annotations']
            new_annos = list()
            new_label = dict()
            index = 0
            for anno in annos:
                if anno['label'] == ocr_target:
                    captions = captions_df[captions_df['Page'] == filename.replace(
                        'json', 'jpg')].reset_index(drop=True)
                    anno['ocr'] = captions.iloc[0]["text_{}".format(index)]
                    anno['rotation'] = int(
                        captions.iloc[0]["rotation_{}".format(index)])
                    index += 1
                new_annos.append(anno)
            new_label['filename'] = filename
            new_label['annotations'] = new_annos
            all_labels.append(new_label)

    with open(output, 'w') as f:
        json.dump(all_labels, f)


#@st.cache_data
def retrieve_cropped_images(labels_path, images_path, cropped_path):
    labels = [os.path.join(labels_path, i) for i in os.listdir(labels_path)]
    for label in labels:
        label = json.load(open(label, 'r'))
        image_name = os.path.join(images_path, label['filename'])
        annos = label['annotations']
        i = 1
        for ann in annos:
            if ann["label"] == "figure":
                image = cv2.imread(image_name)
                pts_list = [[int(pt) for pt in coords]
                            for coords in ann['points']]
                x = [item[0] for item in pts_list]
                y = [item[1] for item in pts_list]
                x1, y1, x2, y2 = min(x), min(y), max(x), max(y)
                cropped_image = image[y1:y2, x1:x2]
                cropped_name = os.path.join(
                    cropped_path, '_'.join([str(i), label["filename"]]))
                i += 1
                cv2.imwrite(cropped_name, cropped_image)


class Preprocessing:
    def __init__(self, images_path):
        self.images_path = images_path
        self.model_path = "model_final.pth"
        self.labels_path = '_'.join([os.path.split(self.images_path)[1], 'labels'])
        self.dataset_name = '_'.join([os.path.split(self.images_path)[1], 'dataset'])
        self.output_path = '_'.join([os.path.split(self.images_path)[1], 'output'])
        self.cropped_path = '_'.join([os.path.split(self.images_path)[1], 'cropped'])
        self.captions_path = '_'.join([os.path.split(self.images_path)[1], 'captions.csv'])
        self.output = '_'.join([os.path.split(self.images_path)[1], 'all_labels.json'])
        os.makedirs(self.labels_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.cropped_path, exist_ok=True)

    def create_data(self):
        create_empty_annotations(self.images_path)

        predict(self.images_path, self.labels_path, self.dataset_name, self.model_path, self.output_path)
        ocr_target = 'caption'
        generate_caption(self.labels_path, self.captions_path, self.images_path, ocr_target)

        #parsing_caption(self.captions_path, self.labels_path, self.output, ocr_target)

        retrieve_cropped_images(self.labels_path,
                                self.images_path, self.cropped_path)
