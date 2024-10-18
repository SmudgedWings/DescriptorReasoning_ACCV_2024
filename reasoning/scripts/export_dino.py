import torch
from torch.utils.data import DataLoader, ConcatDataset
import h5py, os
from glob import glob
from reasoning.datasets.scannet import ScannetSingleH5
from tqdm import tqdm
import argparse
from reasoning.features.dinov2 import DinoV2

possible_models = [
    'dinov2_vits14',
    'dinov2_vitb14',
    'dinov2_vitl14',
    'dinov2_vitg14',
]

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",  "-d", type=str, default="./h5_scannet/", help="Path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=16, help="Batch size")
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14", help="Dino model", choices=possible_models)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DinoV2({"weights": args.dino_model, "allow_resize": True}).to(device)
    model.eval()

    dataset_path = args.dataset
    method_name = "dino-scannet-" + args.dino_model
    features_path = os.path.join(dataset_path, "features", method_name)
    os.makedirs(features_path, exist_ok=True)

    for split in ['train', 'val']:
        all_scenes = glob(os.path.join(dataset_path, split, "*.h5"))
        for scene_h5 in tqdm(all_scenes, desc="Scenes", position=1):
            dataset = ScannetSingleH5(scene_h5, only_images=True)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            h5_fname = os.path.join(features_path, split, os.path.basename(scene_h5))
            os.makedirs(os.path.dirname(h5_fname), exist_ok=True)
            
            with h5py.File(h5_fname, "w") as f:
                f.create_group("features")
                f.create_group("global_descriptor")
                
                for i, data in enumerate(tqdm(dataloader, desc="Extracting features", position=2, leave=False)):
                    with torch.no_grad():
                        images = data["image"].to(device)
                        response = model({"image": images})
                        
                        features = response['features'].cpu().numpy()
                        global_descriptor = response['global_descriptor'].cpu().numpy()
                        
                        for batch_i in range(len(features)):
                            key = data["image_key"][batch_i]
                            f["features"].create_dataset(key, data=features[batch_i])
                            f["global_descriptor"].create_dataset(key, data=global_descriptor[batch_i])