# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import torch
import time

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from detectron2.engine.defaults import DefaultPredictor

from alfred.vis.image.mask import label2color_mask, vis_bitmasks
from alfred.vis.image.det import visualize_det_cv2_part
import numpy as np
from detectron2.data.catalog import MetadataCatalog
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.MODEL.SOLOV2.NMS_PRE = 300
    cfg.MODEL.SOLOV2.SCORE_THR = 0.21
    cfg.MODEL.SOLOV2.UPDATE_THR = 0.25
    cfg.MODEL.SOLOV2.MASK_THR = 0.6
    cfg.MODEL.SOLOV2.MAX_PER_IMG = 150

    cfg.INPUT.MIN_SIZE_TEST = 672  # 90ms
    # cfg.INPUT.MIN_SIZE_TEST = 512 # 70ms
    cfg.INPUT.MIN_SIZE_TEST = 640  # 40ms
    # cfg.INPUT.MAX_SIZE_TEST = 640 # 40ms
    # cfg.INPUT.MAX_SIZE_TEST = 608 # 70ms
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        # nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.65,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def vis_res_fast(res, img, meta):
    # print(meta)
    # print(res)
    ins = res['instances']
    bboxes = ins.pred_boxes.tensor.cpu().numpy()
    scores = ins.scores.cpu().numpy()
    clss = ins.pred_classes.cpu().numpy()
    bit_masks = ins.pred_masks
    # img = vis_bitmasks_with_classes(img, clss, bit_masks)
    img = vis_bitmasks(img, bit_masks)
    img = visualize_det_cv2_part(
        img, scores, clss, bboxes)
    # img = cv2.addWeighted(img, 0.9, m, 0.6, 0.9)
    return img


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    predictor = DefaultPredictor(cfg)

    print(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

    if args.input:
        if os.path.isdir(args.input):
            imgs = glob.glob(os.path.join(args.input, '*.jpg'))
            for path in imgs:
                # use PIL, to be consistent with evaluation
                img = cv2.imread(path)
                tic = time.time()
                res = predictor(img)
                c = time.time() - tic
                print('cost: {}, fps: {}'.format(c, 1/c))
                res = vis_res_fast(res, img, metadata)
                # cv2.imshow('frame', res)
                cv2.imshow('frame', res)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break
        else:
            img = cv2.imread(path)
            tic = time.time()
            res = predictor(img)
            c = time.time() - tic
            print('cost: {}, fps: {}'.format(c, 1/c))
            res = vis_res_fast(res, img, metadata)
            # cv2.imshow('frame', res)
            cv2.imshow('frame', res)
            cv2.waitKey(0)
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        while(video.isOpened()):
            ret, frame = video.read()
            # frame_in = cv2.resize(frame, (640, 640))
            tic = time.time()
            res = predictor(frame)
            c = time.time() - tic
            print('cost: {}, fps: {}'.format(c, 1/c))
            res = vis_res_fast(res, frame, metadata)
            # cv2.imshow('frame', res)
            cv2.imshow('frame', res)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
