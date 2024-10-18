import argparse
import numpy as np
import os
import cv2
import time
from tqdm import tqdm
# import poselib
import json
import multiprocessing as mp
from reasoning.datasets.utils import load_image, to_cv
from reasoning.eval.pose_eval import PoseEval
from functools import partial

import torch
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Disable scientific notation
np.set_printoptions(suppress=True)

def print_fancy(d):
    print(json.dumps(d, indent=2))

def parse():
    parser = argparse.ArgumentParser()

    # Download scannet from LoFTR repo at repohttps://drive.google.com/drive/folders/1nTkK1485FuwqA0DbZrK2Cl0WnXadUZdc
    parser.add_argument("--scannet_path", type=str, help="Path to the Scannet 1500 dataset")
    parser.add_argument("--weights", type=str, help="Weights folder. Ex: \"export_models/xfeat\"")
    parser.add_argument("--output", type=str, default="./output/scannet/", help="Path to the output directory")
    parser.add_argument("--force", action='store_true', help="Force running the benchmark again")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse()
    
    this_dir = os.path.dirname(os.path.abspath(__file__))

    scannet = PoseEval({
        'data_path': args.scannet_path,
        'pairs_path': this_dir + '/scannet1500_pairs_calibrated.txt',
        'output': args.output,
        'ransac_thresholds': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
        'n_workers': 36,
        'cache_images': False,
        'detector_only': False,
    })
    
    functions = {}
    
    from reasoning.features.desc_reasoning import ReasoningBase, Reasoning, load_reasoning_from_checkpoint

    def match_reasoning_model(image0, image1, model):
        with torch.inference_mode():
            response = model.match({
                'image0': image0.unsqueeze(0).to(dev),
                'image1': image1.unsqueeze(0).to(dev),
            })
        
        mkpts0 = response['matches0'][0].detach().cpu().numpy()
        mkpts1 = response['matches1'][0].detach().cpu().numpy()
    
        return mkpts0, mkpts1

    reasoning_model_response = load_reasoning_from_checkpoint(args.weights, "checkpoint_2_1024000.pt")
    model = Reasoning(reasoning_model_response['model']).to(dev).eval()
    match_fn = partial(match_reasoning_model, model=model)

    result = scannet.run_benchmark(matcher_fn = match_fn, name=args.weights, force=args.force)
    print('------------------')
    print('Result')
    print_fancy(result)
