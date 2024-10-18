import torch
from torch.utils.data import DataLoader
import h5py, os
from glob import glob
from reasoning.datasets.scannet import ScannetSingleH5
from tqdm import tqdm
import argparse
import numpy as np

from easy_local_features.feature.baseline_alike import ALIKE_baseline

choices = [
    'alike-t',
    'alike-s',
    'alike-n',
    'alike-l',
]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  "-m", type=str, default="alike-n", choices=choices)
    parser.add_argument("--data",  "-d", type=str, default="./h5_scannet/", help="Path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num_keypoints", "-k", type=int, default=2048, help="Number of keypoints to extract")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    extractor = ALIKE_baseline({
        "model_name":'alike-n',
        "top_k":args.num_keypoints,
        "scores_th":0.0,
        "n_limit":args.num_keypoints,
    })
    extractor.model.eval()
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    extractor.to(device)

    dataset_path = args.data
    method_name =  args.model + "-v2-scannet-n" + str(args.num_keypoints)
    features_path = os.path.join(dataset_path, "features", method_name)
    os.makedirs(features_path, exist_ok=True)

    for split in ['train', 'val']:
        all_scenes = glob(os.path.join(dataset_path, split, "*.h5"))
        for scene_h5 in tqdm(all_scenes, desc="Scenes", position=1):
            dataset = ScannetSingleH5(scene_h5, only_images=True)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
            h5_fname = os.path.join(features_path, split, os.path.basename(scene_h5))
            os.makedirs(os.path.dirname(h5_fname), exist_ok=True)
            
            with h5py.File(h5_fname, "w") as f:
                f.create_group("descriptors")
                f.create_group("keypoints")
                
                for i, data in enumerate(tqdm(dataloader, desc="Extracting features", position=2, leave=False)):
                    with torch.no_grad():
                        images = data["image"].to(device) 
                        for batch_i in range(len(data["image_key"])):
                            key = data["image_key"][batch_i]
                            keypoints, descriptors = extractor.detectAndCompute(images[batch_i])
                            keypoints = keypoints.cpu().numpy()
                            descriptors = descriptors.cpu().numpy()
                            
                            f.create_dataset(f"descriptors/{key}", data=descriptors)
                            f.create_dataset(f"keypoints/{key}", data=keypoints)