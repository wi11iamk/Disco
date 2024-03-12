#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:32:02 2024

@author: wi11iamk
"""

from moviepy.editor import VideoFileClip
import math
import os

# Please ensure that you have the latest version of 'ffmpeg' installed and that it is visible by 'moviepy'.
# This script will manually segment one video into a specified number of videos and save each to a chosen directory.

def segment_video_by_frames(video_path, fps, frames_per_segment, output_dir):
    """
    Segment a video based on a specific number of frames per segment.
    
    Parameters:
    - video_path: Path to the input video file.
    - fps: Frames per second of the video.
    - frames_per_segment: Number of frames per segment.
    - output_dir: Directory to save the segmented videos.
    """
    # Load the video
    video = VideoFileClip(video_path)
    
    # Calculate total number of frames in the video
    total_frames = int(video.fps * video.duration)
    
    # Calculate the number of segments needed
    num_segments = math.ceil(total_frames / frames_per_segment)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Segment the video
    for i in range(num_segments):
        # Calculate the start and end frame for the current segment
        start_frame = i * frames_per_segment
        end_frame = min((i + 1) * frames_per_segment, total_frames)
        
        # Calculate start and end time in seconds for the segment
        start_time = start_frame / fps
        end_time = end_frame / fps
        
        # Define the path for the current segment
        segment_path = os.path.join(output_dir, f"segment_{i+1:03d}.mp4")
        
        # Extract the segment and save it
        segment = video.subclip(start_time, end_time)
        segment.write_videofile(segment_path, codec="libx264", audio_codec="aac")

# Example usage
video_path = '/Users/wi11iamk/Desktop/027_D1.mp4'  # Update this with the path to your video
fps = 120  # Frames per second of your video
frames_per_segment = fps * 10  # For 10-second segments, adjust as needed
output_dir = '/Users/wi11iamk/Desktop/segVideos'  # Output directory

# Please ensure to replace the placeholders with actual values before running
segment_video_by_frames(video_path, fps, frames_per_segment, output_dir)

