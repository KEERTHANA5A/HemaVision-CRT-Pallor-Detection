
# HemaVision

> Automated peripheral perfusion assessment using computer vision, embedded systems, and real-time CRT analysis.

### Automated Peripheral Perfusion Assessment Device using Capillary Refill Time and Skin Pallor Analysis

---

## Why the Name “HemaVision”?

The name **HemaVision** combines:

- **Hema** — derived from *Hematology*, the study of blood and blood-related physiological analysis.
- **Vision** — representing the use of *Computer Vision* techniques for real-time image-based assessment.

Together, the name reflects the project's core objective:
using computer vision and embedded systems to analyze blood perfusion indicators such as Capillary Refill Time (CRT) and skin pallor.

---

## Demo

🎥 **Prototype Demonstration:**  
[Watch HemaVision Demo](result-demo/demo_video_link.md)

---

## Overview

HemaVision is a hardware-software healthcare prototype designed for automated peripheral perfusion assessment using computer vision and embedded systems.

The system measures:

* Capillary Refill Time (CRT)
* Skin Pallor Classification

using real-time fingertip video analysis, pressure standardization, and embedded sensing.

---

## Features

* Real-time CRT detection
* Skin pallor assessment
* Raspberry Pi integration
* OpenCV-based image processing
* Force-sensitive resistor (FSR) pressure detection
* OLED feedback display
* Controlled LED illumination
* Portable embedded healthcare prototype

---

## Tech Stack

### Software

* Python
* OpenCV
* NumPy
* Raspberry Pi OS

### Hardware

* Raspberry Pi Zero 2W
* 5MP Camera Module
* FSR Sensor
* OLED Display
* LED Illumination System
* Custom PCB

---

## Workflow

1. Capture fingertip video
2. Extract ROI
3. Compute red-channel intensity
4. Apply smoothing filter
5. Detect blanching/release
6. Measure CRT recovery
7. Classify pallor level
8. Display results

---

## Research Publication

Author of the research paper:

**“HemaVision: An Automated Peripheral Perfusion Assessment Device Using Capillary Refill Time and Skin Pallor Analysis”**

Developed the complete HemaVision system integrating computer vision, embedded systems, Raspberry Pi–based processing, automated CRT analysis, and skin pallor classification.

**Status:** Accepted for publication in Wiley.

---

## Future Improvements

* Clinical trials
* Adaptive ROI detection
* AI-based pallor prediction
* Temperature compensation
* Cloud integration
* Mobile application support


