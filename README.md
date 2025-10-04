# CSE535 Hand Gesture Recognition Project

This project implements a gesture recognition system that detects and classifies hand gestures from video input. It uses a pre-trained CNN-based feature extractor and cosine similarity for prediction.


## Overview

The goal of this project is to recognize 17 distinct hand gestures, which include:
- Numbers (0–9)
- Home automation gestures like LightOn, FanOff, IncreaseFanSpeed, etc.

The task involves:
- Extracting frames from videos  
- Extracting hand features  
- Mapping them to predefined labels  
- Predicting labels for test videos  

---

## Approach

1. **Frame Extraction**: One frame per video is extracted using a custom function.
2. **Feature Extraction**: A pre-trained handshape feature extractor is used to extract features from frames.
3. **Training**: The model compares training and test features using cosine similarity.
4. **Prediction**: The label with the highest similarity is selected as the prediction.
5. **Output**: All predictions are saved to `Results.csv` in the correct autograder format.


## Directory Structure

