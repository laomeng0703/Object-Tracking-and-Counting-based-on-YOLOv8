# Object-Tracking-and-Counting-based-on-YOLOv8

This is an object tracking and counting project based on the yolov8 model.


### Installation

1. Install the dependencies
```
pip install -r requirements.txt
```

2. Clone & install the repository
```
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .

git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
python setup.py develop
```

### Data preparation

If you want to use your own dataset, make sure to create folders in the root directory with the following structure:
```
data/
├── {name of your dataset}
│   ├── images
│   │   ├── ...
│   ├── label
│   │   ├── ...
```
Then, you need to turn the datasets to yolo format.
If the dataset you are using is in coco format, you can run coco_to_yolo.py.

### Training

After preparing your data set, before starting training, you can download [yolov8 pre-trained weights](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) to the root directory to expect better results.
You can run yolo_train.py to start training.
```
python yolo_train.py
```

### Inference

After train, run yolo_inference.py to get inference.
```
python yolo_inference.py
```

### Tracking & Counting

After you prepare your video and change the video and training weight paths in object_tracking_counting.py, you can start tracking and counting objects.
```
python object_tracking_counting.py
```



