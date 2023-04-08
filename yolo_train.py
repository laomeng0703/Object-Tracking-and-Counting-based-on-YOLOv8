from ultralytics import YOLO


if __name__ == '__main__':
    # load model
    model = YOLO('yolov8x.yaml')
    model = YOLO('yolov8x.pt')
    model = YOLO('yolov8x.yaml').load('yolov8x.pt')

    # train
    model.train(data='pigs.yaml',batch=8, epochs=100, imgsz=640)


