# IMU-Based Gesture Recognition using SVM

Real-time hand gesture recognition from IMU (accelerometer + gyroscope) signals
using Butterworth bandpass filtering and SVM classification, implemented in MATLAB.

## Overview
This project classifies hand gestures from raw IMU sensor data through a signal
processing and machine learning pipeline. Features are extracted from filtered
time-series windows and fed into a trained SVM classifier.

## Pipeline
1. **Data Acquisition** — Raw accelerometer & gyroscope signals from IMU sensor
2. **Preprocessing** — Butterworth bandpass filter to remove noise and drift
3. **Segmentation** — Sliding window segmentation of gesture epochs
4. **Feature Extraction** — Time-domain and frequency-domain features per window
5. **Classification** — SVM with RBF kernel trained per gesture class
6. **Evaluation** — Confusion matrix, accuracy, precision, recall

## Requirements
- MATLAB R2021a or later
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox

