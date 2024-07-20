#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:33:32 2024

@author: wi11iamk
"""

import os
import re
import numpy as np
from parameters import video_info
from moviepy.editor import VideoFileClip, concatenate_videoclips

def measure_drift(drift_points, time_points):
    def drift_function(current_time):
        # Use linear interpolation between the measured drift points
        return np.interp(current_time, time_points, drift_points)
    return drift_function

def process_video(directory, suffix, cut_start_seconds, target_frame_count, condition, drift_points, time_points):
    pattern = re.compile(rf".*{suffix}\.mp4$")
    video_path = None
    
    # Find the video with the specified suffix
    for filename in os.listdir(directory):
        if pattern.match(filename):
            video_path = os.path.join(directory, filename)
            break
    
    if not video_path:
        print(f"No video found with suffix: {suffix}")
        return
    
    # Load the video clip
    clip = VideoFileClip(video_path)
    fps = clip.fps
    
    # Calculate frames to cut from the beginning
    cut_start_frames = cut_start_seconds * fps
    start_time = cut_start_frames / fps
    
    # Target duration is half the video length after cutting the start
    target_duration = (target_frame_count / fps) / 2  # target_frame_count divided by fps gives the duration in seconds
    
    # Initialize variables for dynamic correction
    current_time = start_time
    total_duration_kept = 0
    subclips = []
    keep = True
    
    # Create a drift correction function
    drift_function = measure_drift(drift_points, time_points)
    
    while current_time < clip.duration and total_duration_kept < target_duration:
        correction = drift_function(total_duration_kept)
        adjusted_start_time = current_time + correction
        end_time = adjusted_start_time + 10
        
        if keep:
            # Ensure we do not exceed the clip duration
            end_time = min(end_time, clip.duration)
            subclip = clip.subclip(adjusted_start_time, end_time)
            subclips.append(subclip)
            total_duration_kept += (end_time - adjusted_start_time)
        
        current_time += 10
        keep = not keep
    
    # Concatenate subclips to create the final clip
    final_clip = concatenate_videoclips(subclips)
    
    # Ensure the final clip is the target duration
    if final_clip.duration > target_duration:
        final_clip = final_clip.subclip(0, target_duration)
    
    # Save the new video to the specified directory
    output_directory = f"/Users/wi11iamk/Desktop/PhD/NR/S3/{condition}"
    os.makedirs(output_directory, exist_ok=True)
    output_path = os.path.join(output_directory, f"Afront_{suffix}_drifted.mp4")
    final_clip.write_videofile(output_path, codec='libx264', audio=False)

# Process all videos in the video_info dictionary
directory = "/Users/wi11iamk/Desktop/PhD/NR/S3/Afront"
target_frame_count = 100800  # Target frame count for the new video (14 minutes at 120 fps)
time_points = [60, 210, 410]  

for suffix, (cut_start_seconds, drift_points, condition) in video_info["Afront"].items():
    process_video(directory, suffix, cut_start_seconds, target_frame_count, condition, drift_points, time_points)
