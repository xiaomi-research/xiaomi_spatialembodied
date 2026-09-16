# UPDATE: Replace placeholder paths with your actual paths.
'''
Add meta action as input
'''

import os
import pickle
import json
import numpy as np
import cv2
from tqdm import tqdm
from itertools import chain
import random
import gzip
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
import yaml
import math

def calculate_speed(points_list):
    speeds = []
    for i in range(1, len(points_list)):
        dx = points_list[i][0] - points_list[i-1][0]
        dy = points_list[i][1] - points_list[i-1][1]
        dt = 0.5  # 0.5 seconds interval between points
        vx = dx / dt
        vy = dy / dt
        # speed = ((dx ** 2) + (dy ** 2)) ** 0.5 / dt
        speeds.append([vx,vy])
    return speeds

def calculate_final_displacement(points_list):
    if len(points_list) < 2:
        return 0
    start_point = points_list[0]
    end_point = points_list[-1]
    displacement = ((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2) ** 0.5
    return displacement

def calculate_acceleration(velocity, dt=0.5):
    """
    Calculate acceleration from velocity data

    Args:
        velocity: List of velocity data, each element is a 2D velocity vector [vx, vy]
        dt: Time interval, default is 0.5 seconds

    Returns:
        List of accelerations, each element is a 2D acceleration vector [ax, ay]
    """
    acceleration = []
    for i in range(1, len(velocity)):
        dvx = velocity[i][0] - velocity[i-1][0]
        dvy = velocity[i][1] - velocity[i-1][1]
        ax = dvx / dt
        ay = dvy / dt
        acceleration.append([ax, ay])
    return acceleration

def determine_meta_action(max_speed, max_displacement, avg_acceleration):
    ax, ay = avg_acceleration

    if max_speed < 2 and max_displacement < 1.5:
        return "Keep stationary"
    elif ax > 0.5 or ay > 0.5:
        return "Accelerate"
    elif ax < -0.5 or ay < -0.5:
        return "Decelerate"
    elif -0.5 <= ax <= 0.5 and -0.5 <= ay <= 0.5:
        return "Keep speed"
    else:
        return "Undefined"

def process_data(data):
    jsonlist = []
    for scene in tqdm(scenes):
        files = os.listdir(os.path.join(data_path, scene))
        for file in files:
            gt_path = os.path.join(data_path, scene, file, 'transfuser_target.gz')
            with gzip.open(gt_path, "rb") as f:
                gt_dict = pickle.load(f)

            feat_path = os.path.join(data_path, scene, file, 'ego_status_feature.gz')
            # feat_path = os.path.join(data_path, scene, file, 'transfuser_feature.gz')
            with gzip.open(feat_path, "rb") as f:
                feat_dict = pickle.load(f)

            # get command data
            command = feat_dict['ego_status'][-4:].tolist()
            nav_command = [int(x) for x in command]

            if nav_command == [1,0,0,0]:
                NAV_COMMAND = 'turn left'
            elif nav_command == [0,1,0,0]:
                NAV_COMMAND = 'drive forward'
            elif nav_command == [0,0,1,0]:
                NAV_COMMAND = 'turn right'
            elif nav_command == [0,0,0,1]:
                NAV_COMMAND = 'UNKNOWN'
            else:
                # Any other one-hot layout used to leave NAV_COMMAND unbound and crash
                # with UnboundLocalError when the prompt below was formatted.
                NAV_COMMAND = 'UNKNOWN'

            # get velocity and acceleration
            velocity = [round(x, 4) for x in feat_dict['ego_status'][:2].tolist()]
            acceleration = [round(x, 4) for x in feat_dict['ego_status'][2:4].tolist()]

            # get image info
            image_path = [os.path.join(data_path, scene, file, 'cam_f0.jpg')]

            points_list = gt_dict['trajectory'].tolist()
            speeds = calculate_speed(points_list)
            instantaneous_speeds = []
            for v in speeds:
                vx, vy = v
                speed = math.sqrt(vx**2 + vy**2)
                instantaneous_speeds.append(speed)
            max_speed = max(instantaneous_speeds)
            max_displacement = calculate_final_displacement(points_list)

            accellerate = calculate_acceleration(speeds)
            avg_acceleration = [
                sum(acc[0] for acc in accellerate) / len(accellerate),
                sum(acc[1] for acc in accellerate) / len(accellerate)
            ]
            meta_action = determine_meta_action(max_speed, max_displacement, avg_acceleration)

            # prepare content
            content_user = ""

            content_user = f"""<image>
    Given the following vehicle status:1. Driving Command: {NAV_COMMAND};2. Current velocity: [vx={velocity[0]:.4f}, vy={velocity[1]:.4f}] 3. Current acceleration: [ax={acceleration[0]:.4f}, ay={acceleration[1]:.4f}] 4. Meta Action: {meta_action};
    Task:  Predict 8 future waypoints based on the image and information.
    Requirements: 1.Each waypoint must be in format: [x, y, heading]; 2.Time interval between points: 0.5 seconds;3.Output format must be EXACTLY as follows:[x1,y1,h1], [x2,y2,h2], [x3,y3,h3], [x4,y4,h4], [x5,y5,h5], [x6,y6,h6], [x7,y7,h7], [x8,y8,h8]"""

            formatted_points = []
            for point in points_list:
                formatted_point = [round(x, 4) for x in point]
                formatted_points.append(f"[{formatted_point[0]}, {formatted_point[1]}, {formatted_point[2]}]")

            # trajectory_str = ", ".join(formatted_points)
            content_ass = ", ".join(str([round(x, 4) for x in point]) for point in points_list)

            # content_ass = str([[round(x, 4) for x in point] for point in points_list])

            # Swift framework format
            json_data = {
                "messages": [
                    {
                        "content": content_user,
                        "role": "user"
                    },
                    {
                        "content": content_ass,
                        "role": "assistant"
                    },
                ],
                "images": image_path,
                "meta_action": meta_action,  # Add meta_action to json data
                "token": file
            }

            jsonlist.append(json_data)

    return jsonlist

if __name__ == "__main__":
    data_path = 'data/navsim_navtest_training_cache'
    with open('data/navsim_navtest.yaml', 'r') as f:
        data = yaml.safe_load(f)
    # scenes = os.listdir(data_path)
    scenes = data['log_names']
    num_processes = os.cpu_count()  # Get CPU core count
    print(f"Using {num_processes} processes for parallel processing")

    json_list = process_data(scenes)
    cleaned_list = [item for item in json_list if item is not None]
    json_save_path = "data/navsim_navtest_data_only_traj.json"
    with open(json_save_path, 'w', encoding='utf-8') as json_file:
        json.dump(cleaned_list, json_file, indent=4, ensure_ascii=False)


    print(f"JSON file successfully saved as {json_save_path}")
