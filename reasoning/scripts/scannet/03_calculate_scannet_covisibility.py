from reasoning.datasets.scannet import ScannetImages, add_batch_dim, get_relative_transform, warp_kpts
import torch
import os
import argparse

def calculate_covisibility(scene_folder):

    scannet_images = ScannetImages({
        'data_path': scene_folder,
    })
    
    covis_file = open( os.path.join(scannet_images.root, 'covisibility.txt'), 'w')

    norm_grid = torch.stack(torch.meshgrid(torch.linspace(0, 1, 32), torch.linspace(0, 1, 32)), dim=-1).reshape(-1, 2)

    for source_idx in tqdm(range(len(scannet_images)), position=0, leave=True):
        data_source = add_batch_dim(scannet_images[source_idx])
        
        for target_idx in range(source_idx + 1, len(scannet_images)):            
            data_target = add_batch_dim(scannet_images[target_idx])
            kps0 = (norm_grid * torch.tensor([data_source['image'].shape[3]-1, data_source['image'].shape[2]-1]).float()).unsqueeze(0) # BxNx2
            T_0to1 = get_relative_transform(data_source['pose'], data_target['pose'])
            valid_mask, w_kpts0 = warp_kpts(kps0, data_source['depth'], data_target['depth'], T_0to1, data_source['K'], data_target['K'])
            covis = valid_mask.sum().item() / valid_mask.shape[1]
            covis_file.write(f"{data_source['image_id']},{data_target['image_id']},{covis}\n")
            covis_file.flush()

    covis_file.close()
    
def parse():
    parser = argparse.ArgumentParser(description="Converts ScanNet dataset to H5 format")
    parser.add_argument("--data_path", required=True, help="Path to the ScanNet dataset")
    
    return parser.parse_args()

if __name__ == "__main__":
    from tqdm import tqdm
    args = parse()
    
    all_scenes = sorted(os.listdir(args.data_path + "/scans/"))
    print(f"Found {len(all_scenes)} scenes!")
    for _scene_folder in tqdm(all_scenes):
        # check if it has the 'intrinsics' folder
        if not os.path.exists(os.path.join(args.data_path + "/scans", _scene_folder, 'intrinsics')):
            print(f"{_scene_folder} does not have intrinsics")
            continue
        
        # check if it has the 'covisibility.txt' file
        if os.path.exists(os.path.join(args.data_path + "/scans", _scene_folder, 'covisibility.txt')):
            print(f"{_scene_folder} already has covisibility.txt")
            continue
        
        print(_scene_folder)
        scene_folder = os.path.join(args.data_path + "/scans", _scene_folder)
        calculate_covisibility(scene_folder)
    