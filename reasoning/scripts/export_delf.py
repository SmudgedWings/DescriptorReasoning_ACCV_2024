import torch
from torch.utils.data import DataLoader, ConcatDataset
import h5py, os
from glob import glob
from reasoning.datasets.scannet import ScannetSingleH5
from tqdm import tqdm
import argparse
import numpy as np

from easy_local_features.feature.baseline_superpoint import SuperPoint_baseline
from easy_local_features.feature.baseline_relf import RELF_baseline


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  "-d", type=str, default="./h5_scannet/", help="Path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num_keypoints", "-k", type=int, default=2048, help="Number of keypoints to extract")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    detector = SuperPoint_baseline({
        "top_k":args.num_keypoints,
        "detection_threshold": 0,
        "legacy_sampling": False,
        "top_k": args.num_keypoints,
        "force_num_keypoints": True,
    })
    extractor = RELF_baseline({
        'model': 're_resnet'
    })

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    detector.to(device)
    extractor.to(device)

    dataset_path = args.data
    method_name =  "relf-scannet-n" + str(args.num_keypoints)
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
                            keypoints = detector.detect(images[batch_i].unsqueeze(0))
                            descriptors = extractor.compute(images[batch_i].unsqueeze(0), keypoints)
                            
                            f.create_dataset(f"descriptors/{key}", data=descriptors)
                            f.create_dataset(f"keypoints/{key}", data=keypoints)