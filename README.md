# My personal repo for mission 1 at IMAV 2026 competition

## Overview

The first mission consists of terain mapping, fire and military vehicles detection, description OCR and vehicles GPS coords finding. 

## Reserch

The quick research resulted in several findings:
- the ready dataset for classification specific vehicles at IMAV rather doesn't exist
- the general purpose datasets do exist, so we could identify vehicles
- OCR rather is not as much a problem as getting directly above the text to get good shot
- if we want to make all calculations on the flight we probably can't use really powerful model like YOLOWORLD
- if we want to get classification by model like YOLO8 or YOLO11 we need a dataset
- as we can't create real dataset with French military vehicles, we might try to make a syntethetical one

## Synthetical dataset

- using blender, 3d models of french vehicles and python script and we possibly could get a really good dataset
- this approach was used at IMAV with good results (paper)
- requires learning blender, python scripting in blender and some time with setup

## Current plan

1. Brainstorm the possiblities
2. decide on the solution
3. implement it

## 12/7/2026
Current situation descrition:
- I trained on NOMAD and C2A dataset
- on NOMAD YOLOs with varing results
- first attempt pretty bad -> probably downscalling photos make it impossible for model to learn and predict correctly
- second attempt moderately well (some iterations) -> main idea cut the high resolution photos into pieces and from this pieces make training dataset (P 0.8, R 0.5)
- On C2A set I tried YOLO26n and trained it on kaggle 
- kaggle is go to approach for training probably -> important to use save and commit (works in the backgorund)
- YOLO26n results on C2A (P 0.8, R 0.7)
- hard to say if the C2A dataset that is synthetically made and it's disaster pictures with humans pictures pasted in random places

commands used to train:
```bash
# NOMAD best
yolo task=detect mode=train model=yolov8s.pt data=sliced_data.yaml epochs=150 imgsz=1024 batch=2 mosaic=0.0 copy_paste=0.3 patience=30    
# C2A
!yolo detect train model=yolo26n.pt data=/kaggle/working/data.yaml epochs=100 imgsz=640 batch=32 degrees=90.0 flipud=0.5 fliplr=0.5 project=/kaggle/working/runs name=yolo26_sar_drone
```

Thing to try I think is to train YOLO26 (n or s) on KAGGLE with NOMAD dataset. For this however I need to probably upload NOMAD to kaggle and this in not going to be easy

Uploading NOMAD to kaggle:
- take my cut in pieces images
- upload them to kaggle as zip probably

whole NOMAD -> pretty hard

### IMPORTANT THING
- ask if we run model onboard or on the laptop