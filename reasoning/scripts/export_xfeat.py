import torch
from torch.utils.data import DataLoader
import h5py, os
from glob import glob
from reasoning.datasets.scannet import ScannetSingleH5
from tqdm import tqdm
import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  "-d", type=str, default="./datasets/h5_scannet/", help="Path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--num_keypoints", "-k", type=int, default=2048, help="Number of keypoints to extract")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = args.num_keypoints, detection_threshold=0.)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    dataset_path = args.data
    method_name = "xfeat-scannet-n" + str(args.num_keypoints)
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
                        response = xfeat.detectAndCompute(images, top_k = args.num_keypoints)
                        
                        for batch_i in range(len(response)):
                            key = data["image_key"][batch_i]
                            descriptors = response[batch_i]["descriptors"].cpu().numpy()
                            keypoints = response[batch_i]["keypoints"].cpu().numpy()
                            
                            assert keypoints.shape[0] == args.num_keypoints, f"Expected {args.num_keypoints} keypoints, got {keypoints.shape[0]}"
                            
                            f.create_dataset(f"descriptors/{key}", data=descriptors)
                            f.create_dataset(f"keypoints/{key}", data=keypoints)