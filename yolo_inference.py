from ultralytics import YOLO
from ultralytics.yolo.utils.benchmarks import benchmark

MODEL = 'runs/detect/train5/weights/best.pt'

if __name__ == '__main__':
    # load model
    model = YOLO(MODEL)

    result = model.val(iou=0.65)
    result.box.map
    result.box.map50
    result.box.map75
    result.box.maps

