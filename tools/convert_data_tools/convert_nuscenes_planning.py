# UPDATE: Replace placeholder paths with your actual paths.
import pickle
import os
import json
import re
import argparse
from tqdm import tqdm

def main():
    # ===================== Args: command line parameter configuration =====================
    parser = argparse.ArgumentParser(description="NuScenes dataset: build autonomous driving conversation JSONL")
    # Core: split auto-switch train/val
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'],
                        help="Dataset split: train or val, automatically loads corresponding default files")
    # Optional: manually override paths
    parser.add_argument('--temporal_pkl', type=str, default=None,
                        help="Manually specify nuscenes2d_ego_temporal_infos.pkl path")
    parser.add_argument('--ego_pkl', type=str, default=None,
                        help="Manually specify nuscenes_ego_infos.pkl path")
    parser.add_argument('--drivelmm_json', type=str, default=None,
                        help="Manually specify data_samples.json path")
    # Image root directory
    parser.add_argument('--image_root', type=str,
                        default='data/Nuscenes_Full/samples',
                        help="Image root directory")
    # Output file
    parser.add_argument('--output_path', type=str, required=True,
                        help="Output JSONL file path")

    args = parser.parse_args()

    # ===================== Auto-configure default paths based on split =====================
    split_default_paths = {
        'train': {
            'temporal_pkl': 'data/OmniDrive/data_nusc/nuscenes2d_ego_temporal_infos_train.pkl',
            'ego_pkl': 'data/OmniDrive/omni_pkls/nuscenes_ego_infos_train.pkl',
            'drivelmm_json': 'data/DriveLMM-o1-main/data/finetune/data_samples_train.json'
        },
        'val': {
            'temporal_pkl': 'data/OmniDrive/data_nusc/nuscenes2d_ego_temporal_infos_val.pkl',
            'ego_pkl': 'data/OmniDrive/omni_pkls/nuscenes_ego_infos_val.pkl',
            'drivelmm_json': 'data/DriveLMM-o1-main/data/finetune/data_samples_val.json'
        }
    }

    temporal_pkl = args.temporal_pkl or split_default_paths[args.split]['temporal_pkl']
    ego_pkl = args.ego_pkl or split_default_paths[args.split]['ego_pkl']
    drivelmm_json = args.drivelmm_json or split_default_paths[args.split]['drivelmm_json']

    # ===================== Load data =====================
    with open(ego_pkl, 'rb') as f:
        ego_infos = pickle.load(f)
    print('Loaded ego_infos complete')

    with open(drivelmm_json, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    with open(temporal_pkl, "rb") as f:
        all_samples_omni = pickle.load(f)
    print('All source data loaded')

    # Build token -> sample mapping
    ego_planning_infos = {s['token']: s for s in all_samples}
    ego_planning_infos_omni = {s['token']: s for s in all_samples_omni['infos']}

    # ===================== Count skipped samples =====================
    skip_count = 0
    total_count = len(ego_infos['data_list'])
    # Required Omni data fields (skip if missing)
    required_omni_keys = ['gt_planning', 'gt_fut_traj', 'gt_fut_traj_mask', 'gt_fut_yaw', 'gt_fullnames']

    # ===================== Process frame by frame =====================
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for frame in tqdm(range(total_count)):
            data = ego_infos['data_list'][frame]
            sample_token = data['token']

            # ===================== First check: Token exists =====================
            if sample_token not in ego_planning_infos or sample_token not in ego_planning_infos_omni:
                skip_count += 1
                continue

            omni_sample = ego_planning_infos_omni[sample_token]
            # ===================== Second check: Omni data has all required fields =====================
            if not all(key in omni_sample for key in required_omni_keys):
                skip_count += 1
                continue

            # 1. 6-way camera image paths
            cam_order = [
                'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
            ]
            image_list = [os.path.join(args.image_root, cam, data['images'][cam]['img_path']) for cam in cam_order]

            # 2. Parse ego info
            ego_str = ego_planning_infos[sample_token]['ego']
            info = {}
            # Regex parsing (with null value protection)
            match = re.search(r"Velocity \(vx,vy\): \((.*?),(.*?)\)", ego_str)
            info['vx'] = float(match.group(1).strip()) if match else 0.0
            info['vy'] = float(match.group(2).strip()) if match else 0.0

            match = re.search(r"Heading Angular Velocity \(v_yaw\): \((.*?)\)", ego_str)
            info['v_yaw'] = float(match.group(1).strip()) if match else 0.0

            match = re.search(r"Acceleration \(ax,ay\): \((.*?),(.*?)\)", ego_str)
            info['ax'] = float(match.group(1).strip()) if match else 0.0
            info['ay'] = float(match.group(2).strip()) if match else 0.0

            match = re.search(r"Can Bus: \((.*?),(.*?)\)", ego_str)
            info['can_bus_0'] = float(match.group(1).strip()) if match else 0.0
            info['can_bus_1'] = float(match.group(2).strip()) if match else 0.0

            match = re.search(r"Heading Speed: \((.*?)\)", ego_str)
            info['heading_speed'] = float(match.group(1).strip()) if match else 0.0

            match = re.search(r"Steering: \((.*?)\)", ego_str)
            info['steering'] = float(match.group(1).strip()) if match else 0.0

            match = re.search(r"Mission Goal: (.*?)\n", ego_str)
            info['mission_goal'] = match.group(1).strip() if match else ''

            # 3. Build ego text
            ego_text = (
                f"*****Ego States:*****\n"
                f"Current State:\n"
                f" - Velocity (vx,vy): ({info['vx']:.2f},{info['vy']:.2f})\n"
                f" - Heading Angular Velocity (v_yaw): ({info['v_yaw']:.2f})\n"
                f" - Acceleration (ax,ay): ({info['ax']:.2f},{info['ay']:.2f})\n"
                f" - Can Bus: ({info['can_bus_0']:.2f},{info['can_bus_1']:.2f})\n"
                f" - Heading Speed: ({info['heading_speed']:.2f})\n"
                f" - Steering: ({info['steering']:.2f})\n"
                f"Mission Goal: {info['mission_goal']}"
            )

            # 4. Ground truth planning trajectory
            planning_gt_np = omni_sample['gt_planning'][0][:, :2]
            planning_gt = "[" + ", ".join([f"({x:.6f},{y:.6f})" for x, y in planning_gt_np]) + "]"

            # 5. Other agent ground truth
            gt_fut_traj = omni_sample['gt_fut_traj']
            gt_fut_traj_mask = omni_sample['gt_fut_traj_mask']
            gt_fut_yaw = omni_sample['gt_fut_yaw']
            gt_fullnames = omni_sample['gt_fullnames']

            # Trajectory to string
            def traj_to_str(traj):
                res = []
                for agent_traj in traj:
                    points = [f"({x:.6f}, {y:.6f})" for x, y in agent_traj]
                    res.append(f"[{', '.join(points)}]")
                return "[" + ", ".join(res) + "]"

            gt_fut_traj_str = traj_to_str(gt_fut_traj)
            gt_fut_traj_mask_str = str(gt_fut_traj_mask.tolist())
            gt_fut_yaw_str = str(gt_fut_yaw.tolist())
            gt_fullnames_str = str(gt_fullnames)

            # 6. Build messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an driving assistant that makes autonomous driving decisions by observing surround-view images and utilizing vehicle state information. Your task is to generate safe, accurate and reasonable future trajectory planning for the ego-vehicle."
                },
                {
                    "role": "user",
                    "content": f"<image>(front), <image>(front right), <image>(front left), <image>(back), <image>(back left), <image>(back right) are the images captured by the ego-vehicle's surround-view cameras. The current state of the ego-vehicle is as follows:\n{ego_text}\nPlease generate the future 3-second decision trajectory of the ego-vehicle (6 frames at 2Hz sampling rate).\nYou must output only the trajectory in the format: [(x1,y1), (x2,y2), (x3,y3), (x4,y4), (x5,y5), (x6,y6)]. DO NOT output any extra text, explanation, or symbols."
                },
                {
                    "role": "assistant",
                    "content": planning_gt
                }
            ]

            # 7. Write file
            final_sample = {
                "messages": messages,
                "images": image_list,
                "metadata": {
                    "token": sample_token,
                    "gt_fut_traj": gt_fut_traj_str,
                    "gt_fut_traj_mask": gt_fut_traj_mask_str,
                    "gt_fut_yaw": gt_fut_yaw_str,
                    "gt_fullnames": gt_fullnames_str,
                }}
            f_out.write(json.dumps(final_sample, ensure_ascii=False) + '\n')

    print(f"\nProcessing complete!")
    print(f"Total samples: {total_count} | Valid samples: {total_count - skip_count} | Skipped (missing data): {skip_count}")
    print(f"Output file: {args.output_path}")

if __name__ == '__main__':
    main()
