#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:04:13 2024

@author: wi11iamk
"""

import os
import re
import numpy as np
from moviepy.editor import VideoFileClip, concatenate_videoclips
from send2trash import send2trash
from PIL import Image

def custom_resize(clip, newsize):
    def resize_image(image):
        pil_image = Image.fromarray(image)
        resized_pil_image = pil_image.resize(newsize, Image.LANCZOS)
        return np.array(resized_pil_image)
    
    return clip.fl_image(resize_image)

def find_videos(directory, suffix):
    videos = []
    pattern = re.compile(r"([A-Za-z0-9]+)([0-9]{3})\.MP4$")
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            print(f"Found file: {filename}, matched suffix: {match.group(2)}")
        if match and match.group(2) == suffix:
            videos.append(os.path.join(directory, filename))
    return videos

def stitch_videos(video1_path, video2_path, output_path, resolution):
    # Load the video clips
    clip1 = VideoFileClip(video1_path)
    clip2 = VideoFileClip(video2_path)
    
    # Ensure both clips have the same FPS
    fps = clip1.fps
    
    # Resize clips to the specified resolution using custom resize function
    clip1 = custom_resize(clip1, resolution)
    clip2 = custom_resize(clip2, resolution)
    
    # Determine the order based on duration
    if clip1.duration < clip2.duration:
        clip1, clip2 = clip2, clip1
    
    # Concatenate the video clips
    final_clip = concatenate_videoclips([clip1, clip2])
    
    # Write the output video file with the specified FPS
    final_clip.write_videofile(output_path, fps=fps, codec='libx264', audio_codec='aac')
    
    # Move original videos to bin
    send2trash(video1_path)
    send2trash(video2_path)

def find_all_suffixes(directory):
    suffixes = set()
    pattern = re.compile(r"([A-Za-z0-9]+)([0-9]{3})\.MP4$")
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            suffixes.add(match.group(2))
    return suffixes

# Example usage
directory = "/Users/wi11iamk/Desktop/PhD/NR/S3/Aside"
output_directory = "/Users/wi11iamk/Desktop/PhD/NR/S3/AsideNew"
resolution = (1024, 576)  # Desired resolution (width, height)

# Find all unique suffixes in the directory
suffixes = find_all_suffixes(directory)

for suffix in suffixes:
    print(f"Processing suffix: {suffix}")
    # Find videos with the specified suffix
    videos = find_videos(directory, suffix)
    
    if len(videos) == 2:
        video1_path, video2_path = videos
        print(f"Found videos: {video1_path} and {video2_path}")
        output_path = os.path.join(output_directory, f"NEW_{suffix}.mp4")
        stitch_videos(video1_path, video2_path, output_path, resolution)
    else:
        print(f"Could not find exactly two videos with the specified suffix: {suffix}.")