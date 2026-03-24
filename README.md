# Polaris: Real-Time Object Detection System

A high-performance, real-time object detection and tracking system built using Ultralytics YOLO, OpenCV, and Python. The system is designed to process live video streams with low latency while maintaining reliable detection accuracy.

---
## Demo

This system processes video in real time, performing object detection and tracking with a focus on performance and reliability.

[Watch Demo](https://www.youtube.com/watch?v=TL304i3NAz4)

## Overview

Polaris is an end-to-end computer vision pipeline capable of detecting and tracking objects in real time from both video files and live camera feeds. The system emphasizes performance, modularity, and scalability for real-world applications.

---

## Features

- Real-time object detection using YOLO  
- Support for live camera input and video files  
- Object tracking integration (e.g., ByteTrack or BoT-SORT)  
- Optimized for low latency and stable frame rates  
- Modular and extensible code structure  
- Compatible with CPU and GPU environments  

---

## Tech Stack

- Python  
- Ultralytics YOLO  
- OpenCV  
- PyTorch  
- NumPy  

---

## System Design

The system follows a modular pipeline:

- Frame Capture: Reads input from video or camera  
- Inference: Performs object detection using YOLO  
- Tracking: Assigns consistent IDs to objects across frames  
- Rendering: Displays annotated output in real time  

This structure ensures smooth performance and supports future extensions.

---

## Why "Polaris"

Polaris, the North Star, has long been used as a stable point of reference for navigation. This project follows the same principle—providing consistent, real-time understanding of dynamic visual environments.

The system is designed to reliably detect and track objects under changing conditions, acting as a stable reference point within motion-heavy scenes. The name reflects a focus on building systems that are dependable, clear, and effective in real-world applications.
