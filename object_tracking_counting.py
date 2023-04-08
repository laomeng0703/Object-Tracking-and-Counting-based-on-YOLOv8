import os
import sys

import torch
from ByteTrack import yolox
from ultralytics import YOLO

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass

import supervision
from typing import List

from supervision.draw.color import ColorPalette
from supervision.geometry.dataclasses import Point
from supervision.video.dataclasses import VideoInfo
from supervision.video.source import get_video_frames_generator
from supervision.video.sink import VideoSink
from supervision.tools.detections import Detections, BoxAnnotator

from line_counter import LineCount, LineAnnotator

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#----------------------------settings-------------------------------#
'''Change your own settings'''
LINE_START = Point(639, 0)
LINE_END = Point(641, 720)

MODEL = 'runs/detect/train5/weights/best.pt'

TARGET_VIDEO_PATH = 'coding_test_dataset/track_and_count_result.mp4'
SOURCE_VIDEO_PATH = 'coding_test_dataset/inference_video.mp4'


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float         = 0.25
    track_buffer: int           = 30
    match_thresh: float         = 0.8
    aspect_ratio_thresh: float  = 3.0
    min_box_area: float         = 1.0
    mot20: bool                 = False

#------------------------------------------Tracking Utils------------------------------------------#
# manually match the bounding boxes coming from our model with those created by the tracker.
#--------------------------------------------------------------------------------------------------#
# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches bounding boxes with predictions
def match_detections_with_tracks(
        detections: Detections,
        tracks: List[STrack]) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids
#------------------------------------------------------------------------------------------------#

#-------------------------------------predict & annotate-----------------------------------------#
# load model
model = YOLO(MODEL)
model.fuse()

# dict maping class_id to class_name
CLASS_NAMES_DICT = model.names
# class_ids of interest - pig
CLASS_ID = [0]

# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)
# create LineCount instance
line_count = LineCount(start=LINE_START, end=LINE_END)
# create instance of BoxAnnotator and LineAnnotator
box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.3)
line_annotator = LineAnnotator(thickness=1, text_thickness=1, text_scale=0.5)

# open target video file
with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    # loop over video frames
    for frame in tqdm(generator, total=video_info.total_frames):

        # model prediction on single frame and conversion to supervision Detections
        results = model(frame)
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int)
        )

        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # format custom labels
        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        # updating line counter
        line_count.update(detections=detections)

        # annotate and display frame
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        line_annotator.annotate(frame=frame, line_counter=line_count)
        sink.write_frame(frame=np.uint8(frame))
