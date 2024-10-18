from reasoning.datasets.scannet import makeScannetDataset
import torch
import os
import argparse
import h5py
from tqdm import tqdm

def parse():
    parser = argparse.ArgumentParser(description="Converts ScanNet dataset to H5 format")
    parser.add_argument("--data_path", required=True, help="Path to the ScanNet dataset")
    parser.add_argument("--output", required=True, help="Path to the output directory for H5 files")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    args = parse()
    assert os.path.exists(args.data_path)
    os.makedirs(args.output, exist_ok=True)

    # splits = ['test', 'val', 'train']
    splits = ['train']

    for split in splits:
        pairs = makeScannetDataset(os.path.join(args.data_path, "scans"), 0.4, 0.8, split, concat=False)
        for scene_idx, (scene,dataset) in enumerate(pairs.items()):
            
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=16)
            h5_fname = f"{args.output}/{split}/{scene}.h5"
            os.makedirs(os.path.dirname(h5_fname), exist_ok=True)

            with h5py.File(h5_fname, 'w') as f:
                # create group for images
                images_group = f.create_group('images') # key is image_key
                depth_group = f.create_group('depth') # key is image_key
                pose_group = f.create_group('pose') # key is image_key
                K_group = f.create_group('K') # key is image_key
                T_0to1_group = f.create_group('T_0to1') # key is image_key0:image_key1
                T_1to0_group = f.create_group('T_1to0') # key is image_key0:image_key1
                covis_group = f.create_group('covis') # key is image_key0:image_key1
                
                for i, data in enumerate(tqdm(dataloader, desc=f"Processing {scene} {split} ({scene_idx+1}/{len(pairs)}")):
                    image0 = data['image0']
                    image1 = data['image1']
                    depth0 = data['depth0']
                    depth1 = data['depth1']
                    pose0 = data['pose0']
                    pose1 = data['pose1']
                    T_0to1 = data['T_0to1']
                    T_1to0 = data['T_1to0']
                    # K = data['K']
                    K0 = data['K0']
                    K1 = data['K1']
                    covis = data['covis']
                    dataset_name = data['dataset_name']
                    image0_key = data['image0_key']
                    image1_key = data['image1_key']
                    
                    B = len(image0_key)
                    
                    for b in range(B):
                        # write to h5 
                        key0 = image0_key[b].replace("/", "_")
                        key1 = image1_key[b].replace("/", "_")
                        
                        if key0 not in images_group:
                            images_group.create_dataset(key0, data=image0[b].numpy())
                            depth_group.create_dataset(key0, data=depth0[b].numpy())
                            pose_group.create_dataset(key0, data=pose0[b].numpy())
                            K_group.create_dataset(key0, data=K0[b].numpy())
                            
                        if key1 not in images_group:
                            images_group.create_dataset(key1, data=image1[b].numpy())
                            depth_group.create_dataset(key1, data=depth1[b].numpy())
                            pose_group.create_dataset(key1, data=pose1[b].numpy())
                            K_group.create_dataset(key1, data=K1[b].numpy())
                            
                        covis_group.create_dataset(f"{key0}:{key1}", data=covis[b])
                        T_0to1_group.create_dataset(f"{key0}:{key1}", data=T_0to1[b].numpy())
                        T_1to0_group.create_dataset(f"{key0}:{key1}", data=T_1to0[b].numpy())