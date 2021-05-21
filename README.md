# SoloV2

This project based on Adelaidet, but we removed all files unrelated with SOLOv2 for simplicity. The purpose of this repo are:

- [ ] Training cityscapes && custom dataset;
- [ ] Try different backbone for instance segmentation;
- [ ] Experiment on different SxS settings to **improve** small object accuracy;
- [x] ONNX export;
- [ ]  TensorRT deployment;



## Demo

Quick demo:

```
python3 demo/demo_fast.py --config-file configs/SOLOv2/R50_3x.yaml --video-input /media/jintian/samsung/datasets/public/TestVideos/road_demo.mp4 --opts MODEL.WEIGHTS weights/SOLOv2_R50_3x.pth
```

