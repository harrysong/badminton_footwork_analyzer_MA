"""
Configuration file for Badminton Footwork Analyzer
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure directories exist
for dir_path in [DATA_DIR, VIDEO_DIR, REFERENCE_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# MediaPipe Pose configuration
MEDIAPIPE_POSE_CONFIG = {
    "static_image_mode": False,
    "model_complexity": 1,  # 0, 1, or 2 (2 is most accurate but slowest)
    "smooth_landmarks": True,
    "enable_segmentation": False,
    "smooth_segmentation": False,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# Body segment weights for Center of Mass calculation
# Based on anthropometric data (Winter, 2009)
BODY_WEIGHTS = {
    "head": 0.081,
    "neck": 0.028,
    "torso_upper": 0.166,
    "torso_mid": 0.116,
    "torso_lower": 0.146,
    "upper_arm": 0.028,  # each
    "forearm": 0.016,  # each
    "hand": 0.006,  # each
    "thigh": 0.100,  # each
    "shin": 0.047,  # each
    "foot": 0.014,  # each
}

# MediaPipe landmark indices
LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# CoM calculation landmark groups
COM_LANDMARK_GROUPS = {
    "head": ["nose"],
    "torso_upper": ["left_shoulder", "right_shoulder"],
    "torso_mid": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    "torso_lower": ["left_hip", "right_hip"],
    "left_upper_arm": ["left_shoulder", "left_elbow"],
    "left_forearm": ["left_elbow", "left_wrist"],
    "left_hand": ["left_wrist"],
    "right_upper_arm": ["right_shoulder", "right_elbow"],
    "right_forearm": ["right_elbow", "right_wrist"],
    "right_hand": ["right_wrist"],
    "left_thigh": ["left_hip", "left_knee"],
    "left_shin": ["left_knee", "left_ankle"],
    "left_foot": ["left_ankle", "left_heel", "left_foot_index"],
    "right_thigh": ["right_hip", "right_knee"],
    "right_shin": ["right_knee", "right_ankle"],
    "right_foot": ["right_ankle", "right_heel", "right_foot_index"],
}

# Video processing
VIDEO_CONFIG = {
    "target_fps": 30,
    "max_resolution": (1920, 1080),
    "output_format": "mp4",
    "output_codec": "mp4v",
}

# Analysis parameters
ANALYSIS_CONFIG = {
    # Movement detection
    "movement_threshold": 5.0,  # pixels/frame
    "acceleration_threshold": 2.0,  # pixels/frame^2

    # Footwork event detection
    "jump_threshold": 10,  # pixels y-axis movement
    "landing_velocity_threshold": -20,  # pixels/frame
    "direction_change_threshold": 90,  # degrees

    # Smoothing
    "kalman_process_noise": 1e-5,
    "kalman_measurement_noise": 1e-2,

    # Heatmap
    "heatmap_grid_size": (50, 50),  # court grid
    "heatmap_bandwidth": 0.5,

    # Efficiency metrics
    "path_efficiency_window": 30,  # frames
    "response_time_window": 15,  # frames (0.5s at 30fps)
}

# Court dimensions (badminton singles in meters)
COURT_CONFIG = {
    "length": 13.4,
    "width": 5.18,
    "service_line_offset": 1.98,
    "pixel_to_meter": None,  # Will be calibrated from video
}

# Visualization
VIZ_CONFIG = {
    "skeleton_color": (0, 255, 0),
    "com_color": (0, 0, 255),
    "trajectory_color": (255, 0, 0),
    "heatmap_alpha": 0.5,
    "trajectory_length": 30,  # frames to show
    "line_thickness": 2,
    "keypoint_radius": 3,
}

# Dashboard
DASHBOARD_CONFIG = {
    "max_video_width": 800,
    "chart_height": 400,
    "update_interval": 0.1,  # seconds
}
