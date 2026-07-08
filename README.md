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