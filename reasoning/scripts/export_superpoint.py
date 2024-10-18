import torch
from torch.utils.data import DataLoader, ConcatDataset
import h5py, os
from glob import glob
from reasoning.datasets.scannet import ScannetSingleH5
from tqdm import tqdm
import argparse

from reasoning.features.superpoint import SuperPoint

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  "-d", type=str, default="./h5_scannet/", help="Path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num_keypoints", "-k", type=int, default=2048, help="Number of keypoints to extract")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    extractor = SuperPoint({
        "has_detector": True,
        "has_descriptor": True,
        "descriptor_dim": 256,
        # Inference
        "sparse_outputs": True,
        "dense_outputs": False,
        "nms_radius": 4,
        "refinement_radius": 0,
        "detection_threshold": 0.,
        "max_num_keypoints": args.num_keypoints,
        "max_num_keypoints_val": None,
        "force_num_keypoints": True,
        "randomize_keypoints_training": False,
        "remove_borders": 4,
        "legacy_sampling": False,  # True to use the old broken sampling
    })
    extractor = extractor.cuda()
    extractor.eval()
    
    dataset_path = args.data
    method_name = f"superpoint-scannet-n" + str(args.num_keypoints)
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
                        images = data["image"].cuda() 
                        response = extractor({
                            "image": images,
                        })
                        for batch_i in range(len(response["keypoints"])):
                            key = data["image_key"][batch_i]
                            descriptors = response["descriptors"][batch_i].cpu().numpy()
                            keypoints = response["keypoints"][batch_i].cpu().numpy()
                            
                            f.create_dataset(f"descriptors/{key}", data=descriptors)
                            f.create_dataset(f"keypoints/{key}", data=keypoints)