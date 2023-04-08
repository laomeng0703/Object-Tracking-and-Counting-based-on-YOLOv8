"""
COCO 格式的数据集转化为 YOLO 格式的数据集
--json_path 输入的json文件路径
--save_path 保存的文件夹名字，默认为当前目录下的labels。
"""

import os
import json
from tqdm import tqdm
import argparse
import codecs
import shutil
from glob import glob
import codecs
import cv2

label_path = 'coding_test_dataset/json/'
save_path = 'datasets/'

if not os.path.exists(save_path + "pig/label/train2023/"):
    os.makedirs(save_path + "pig/label/train2023/")
if not os.path.exists(save_path + "pig/images/train2023/"):
    os.makedirs(save_path + "pig/images/train2023/")

txt_save_path = 'datasets/pig/label/train2023/'

files = glob(label_path + "*.json")
files = [i.replace("\\", "/").split("/")[-1].split(".json")[0] for i in files]
print(files)

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]

    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


if __name__ == '__main__':
    for json_file_ in tqdm(files):
        json_filename = label_path + json_file_ + ".json"
        json_file = json.load(open(json_filename, "r", encoding="utf-8"))
        height, width, channels = cv2.imread('coding_test_dataset/image/' + json_file_ + ".jpg").shape

        id_map = {}  # coco数据集的id不连续！重新映射一下再输出！
        with codecs.open(txt_save_path + json_file_ + '.txt', 'w', 'utf-8') as f:
            # 写入classes.txt
            for i, category in enumerate(json_file['categories']):
                id_map[category['id']] = i
            # print(id_map)

            anns = {}
            for ann in json_file["annotations"]:
                imgid = ann["image_id"]
                anns.setdefault(imgid, []).append(ann)

            # print('got anns')

            # for img in json_file["images"]:
            #     filename = img["file_name"]
            #     img_width = img["width"]
            #     img_height = img["height"]
            #     img_id = img["id"]
            #     # head, tail = os.path.splitext(filename)

            # ann_img = anns.get(img_id, [])
            # for ann in ann_img:
                box = convert((width, height), ann["bbox"])
                f.write("%s %s %s %s %s\n" % (id_map[ann["category_id"]], box[0], box[1], box[2], box[3]))
            f.close()

image_files = glob('coding_test_dataset/image/' + "*.jpg")
print('copy image files to datasets/pig/images/train2023/')
for image in image_files:
    shutil.copy(image, save_path + 'pig/images/train2023/')