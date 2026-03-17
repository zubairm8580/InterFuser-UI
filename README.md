# InterFuser UI

> Real-time autonomous driving monitoring interface powered by [InterFuser](https://github.com/opendilab/InterFuser) and [CARLA Simulator](https://carla.org/).

![InterFuser UI Demo](assets/demo.gif)

## Demo Video

[![InterFuser UI Demo](https://img.youtube.com/vi/BwOTBmsP8jI/0.jpg)](https://www.youtube.com/watch?v=BwOTBmsP8jI)

## Overview

This project provides a **Pygame-based real-time monitoring UI** for the InterFuser autonomous driving model running inside the CARLA simulator. It visualizes the model's perception, decision-making, and control outputs in a three-panel layout.

### UI Layout (1600 x 720)

| Left Panel (350px) | Center Panel (900px) | Right Panel (350px) |
|---|---|---|
| Speed & Controls | Rear Camera (2K) | Front Camera |
| Decision Reasoning | | Left Camera |
| LiDAR BEV (Radar) | | Right Camera |

### Features

- **InterFuser Model Inference** - 4 RGB cameras + LiDAR input, real-time waypoint prediction
- **Safety Rules**
  - CARLA ground-truth traffic light detection (Red & Yellow)
  - Model-based red light detection
  - Stop sign detection with timed resume (3s wait + 15s cooldown)
  - Startup grace period (brake hold for model stabilization)
  - Max speed limit (40 km/h)
- **PID Control** - Steering and speed control matching original InterFuser implementation
- **Route Planning** - CARLA GlobalRoutePlanner-based navigation with automatic re-routing
- **LiDAR BEV** - Radar-style bird's eye view visualization

## Project Structure

```
InterFuser-UI/
├── UI.py                  # Main application (~1000 lines)
├── interfuser_core/       # InterFuser model core (adapted from original repo)
│   └── timm/              # Custom timm modules for InterFuser
├── requirements.txt       # Python dependencies
├── assets/
│   └── demo.gif           # Demo animation
├── CARLA_0.9.16/          # (Not included) CARLA simulator - download separately
└── checkpoints/           # (Not included) Model weights - download separately
    └── interfuser.pth
```

## Requirements

### Hardware (Tested On)

| Component | Spec |
|---|---|
| CPU | Intel Core i9-14900K |
| RAM | 128 GB |
| GPU | NVIDIA RTX 5090 |

### Software

- **Python** 3.12
- **CARLA** 0.9.16
- **PyTorch** 2.10.0+ (CUDA 12.8)

## Installation

### 1. CARLA Simulator

Download [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases/tag/0.9.16/) and extract it to the project root as `CARLA_0.9.16/`.

### 2. Model Checkpoint

Download the pretrained InterFuser checkpoint from the [original repository](https://github.com/opendilab/InterFuser) and place it at `checkpoints/interfuser.pth`.

### 3. Python Environment

```bash
# Create conda environment
conda create -n interfuser python=3.12 -y
conda activate interfuser

# Install CARLA Python API
pip install CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-win_amd64.whl

# Install PyTorch with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

## Usage

```bash
# 1. Start CARLA server
./CARLA_0.9.16/CarlaUE4.exe

# 2. Run InterFuser UI
conda activate interfuser
python UI.py
```

Press `ESC` to quit.

## Acknowledgements

This project builds upon the excellent work of the InterFuser team. The model architecture and core modules in `interfuser_core/` are adapted from the original implementation.

> **InterFuser: Safety-Enhanced Autonomous Driving Using Interpretable Sensor Fusion Transformer**
> Hao Shao, Letian Wang, RuoBing Chen, Hongsheng Li, Yu Liu
> [Paper (CoRL 2023)](https://arxiv.org/abs/2207.14024) | [Original Repository](https://github.com/opendilab/InterFuser)

Thank you to the original authors for making their research and code publicly available, enabling projects like this to be built on top of their work.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file.
