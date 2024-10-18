from reasoning.datasets.scannet import SensorData
from glob import glob
import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str, help='Path to the data directory')
    args = parser.parse_args()
    return args.data_path

if __name__ == "__main__":
    data_path = parse()
    # find .sens files recursively
    sens_files = glob(os.path.join(data_path, "scans", "**", "*.sens"), recursive=True)
    
    for i, sens_file in enumerate(sens_files):
        print(f"Processing {i+1}/{len(sens_files)}: {sens_file}")
        # if we already processed this file, skip it
        
        folder = os.path.dirname(sens_file)
        if os.path.exists(folder + "/intrinsics"):
            print("Already processed, skipping")
            continue
        
        sensor_data = SensorData(sens_file)
        sensor_data.export_color_images(output_path=folder + "/color", frame_skip=15)
        sensor_data.export_depth_images(output_path=folder + "/depth", frame_skip=15)
        sensor_data.export_poses(output_path=folder + "/poses", frame_skip=15)
        sensor_data.export_intrinsics(output_path=folder + "/intrinsics")