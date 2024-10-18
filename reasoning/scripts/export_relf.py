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
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size")
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

    device = torch.device("cuda")#if torch.cuda.is_available() else torch.device("cpu")
    detector.to(device)
    extractor.to(device)

    dataset_path = args.data
    method_name =  "relf-scannet-n" + str(args.num_keypoints)
    features_path = os.path.join(dataset_path, "features", method_name)
    os.makedirs(features_path, exist_ok=True)

    for split in ['train', 'val']:
        all_scenes = glob(os.path.join(dataset_path, split, "*.h5"))
        scene_idx = 0
        for scene_h5 in tqdm(all_scenes, desc="Scenes", position=1):
            # if scene_idx < 305:
            #     scene_idx += 1
            #     continue
            
            dataset = ScannetSingleH5(scene_h5, only_images=True)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
            h5_fname = os.path.join(features_path, split, os.path.basename(scene_h5))
            os.makedirs(os.path.dirname(h5_fname), exist_ok=True)
            
            # if os.path.exists(h5_fname):
            #     f = h5py.File(h5_fname, "a")
            # else:
            f = h5py.File(h5_fname, "w")

            if not "descriptors" in f:
                f.create_group("descriptors")
                
            if not "keypoints" in f:
                f.create_group("keypoints")
                
            # keys = list(f["descriptors"].keys())

            for i, data in enumerate(tqdm(dataloader, desc="Extracting features", position=2, leave=False)):
                # data_keys = list(data["image_key"])
                # all_keys_already_done = [True if key in keys else False for key in data_keys]
                # if all(all_keys_already_done):
                #     continue
                
                with torch.no_grad():
                    images = data["image"].to(device) 
                    bkeypoints = detector.detect(images)
                    bkeypoints, bdescriptors = extractor.compute(images, bkeypoints)
                    
                    for batch_i in range(len(bkeypoints)):
                        key = data["image_key"][batch_i]
                        descriptors = bdescriptors[batch_i].cpu().numpy()
                        keypoints = bkeypoints[batch_i].cpu().numpy()
                        
                        assert keypoints.shape[0] == args.num_keypoints, f"Expected {args.num_keypoints} keypoints, got {keypoints.shape[0]}"
                        
                        f.create_dataset(f"descriptors/{key}", data=descriptors)
                        f.create_dataset(f"keypoints/{key}", data=keypoints)
            scene_idx += 1
            f.close()