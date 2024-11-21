import torch
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from reasoning.datasets.scannet import ScannetH5
from reasoning.datasets.utils import batch_to_device

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from omegaconf import OmegaConf

from reasoning.datasets.utils import batch_to_device, to_cv
from reasoning.modules.visualization import plot_matches, plot_keypoints, plot_pair
import cv2
from matplotlib import pyplot as plt
import poselib
from tqdm import tqdm

def next_free_port( port=10024, max_port=65535 ):
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    print(f"[GPU{rank}] Setting up DDP at {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}")
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
        args
    ) -> None:

        self.args = args
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        if not args.local:
            self.raw_model = self.model = model.to(gpu_id)
            self.model = DDP(self.raw_model, device_ids=[gpu_id], gradient_as_bucket_view=True, find_unused_parameters=True)
            self.raw_model = self.model.module
        else:
            print("[W] -> Training locally! No DPP.")
            self.gpu_id = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.raw_model = self.model = model
            if not torch.cuda.is_available():
                print("[W] -> Training on CPU! You probably dont want this, but could be usefull for debug.") 
        
        # create tensorboard writer in the first process
        self.global_iter = 0
        self.global_batch = 0
        self.global_epoch = 0
        self.world_size = 1 if not torch.cuda.is_available() else torch.cuda.device_count()
        if gpu_id == 0:
            self.scalars = {}
            
            comment = ''
            if args.debug:
                comment += '-debug'

            if args.comment:
                comment += f"-{args.comment}"
            
            self.writer = SummaryWriter(comment=comment)
            self.model_folder = self.writer.log_dir
            
            with open(f"{self.model_folder}/model_config.yaml", "w") as f:
                OmegaConf.save(self.raw_model.conf, f)
                
            # save all args
            with open(f"{self.model_folder}/args.yaml", "w") as f:
                OmegaConf.save(args, f)
                
        if args.resume:
            self._load_checkpoint(args.resume)
    
    def plot_qualitatives(self, data, pred):
        with torch.inference_mode():
            res = self.raw_model.match(data, pred)
            batch_mkpts0, batch_mkpts1 = res['matches0'], res['matches1']
            
                                    
        #####################################################################################
        for b_index in range(min(self.args.batch_size, 5)):                        
            # plot a match
            img0 = to_cv(data['image0'][b_index], to_gray=True)
            img1 = to_cv(data['image1'][b_index], to_gray=True)
            
            f, a = plot_pair(img0, img1, figsize=(10, 5))
            plot_keypoints(pred['keypoints0'][b_index], pred['keypoints1'][b_index], kps_size=3, edgecolor='black')

            mkpts0 = batch_mkpts0[b_index].detach().cpu().numpy()
            mkpts1 = batch_mkpts1[b_index].detach().cpu().numpy()
            
            if 'scale0' in data:
                scaled_mkpts0 = (torch.tensor(mkpts0, device=self.gpu_id) * data['scale0'][b_index]).detach().cpu().numpy()
                scaled_mkpts1 = (torch.tensor(mkpts1, device=self.gpu_id) * data['scale1'][b_index]).detach().cpu().numpy()
            else:
                scaled_mkpts0 = mkpts0
                scaled_mkpts1 = mkpts1

            if len(scaled_mkpts0) > 0:
                if len(scaled_mkpts0) > 8:
                    F, info = poselib.estimate_fundamental(scaled_mkpts0, scaled_mkpts1, {'max_epipolar_error': 2.0}, {})
                    inliers = info['inliers']
                else:
                    inliers = np.ones(len(scaled_mkpts0)).astype(bool)

                inliers_colors = ['g' if c else 'r' for c in inliers]
                plot_matches(mkpts0, mkpts1, color=inliers_colors, alpha=0.4)
                a[0].text(0, 0, f"Matches {len(mkpts0)} Inliers {sum(inliers)} ", fontsize=12, color='white', backgroundcolor='black')

            plt.tight_layout()
            try:
                self.writer.add_figure(f"_matches_{b_index}/Reasoning", f, self.global_iter)
                plt.close('all')
            except:
                print("Error adding figure")
    
    def add_scalars(self, name, scalar):
        if name not in self.scalars:
            self.scalars[name] = []
        
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.item()
        
        if isinstance(scalar, float):
            self.scalars[name].append(scalar)
        elif isinstance(scalar, dict):
            for k, v in scalar.items():
                self.add_scalars(f"{name}/{k}", v)
        else:
            raise ValueError(f"Unknown scalar type {type(scalar)}")

    def log_all(self):
        for name, scalar in self.scalars.items():
            scalar = [s for s in scalar if s is not None]
            if len(scalar) > 0:
                self.writer.add_scalar(name, np.nanmean(scalar), self.global_iter)
        self.scalars = {}
        self.writer.add_scalar("0_train/epoch", self.global_epoch, self.global_iter)
        self.writer.flush()    

    def _run_batch(self, data):
        data = batch_to_device(data, self.gpu_id)
        pred0 = self.model({
            'image': data['image0'],
            'semantic_features': data['semantic_features0'],
            'descriptors': data['descriptors0'],
            'keypoints': data['keypoints0'],
        })
        pred1 = self.model({
            'image': data['image1'],
            'semantic_features': data['semantic_features1'],
            'descriptors': data['descriptors1'],
            'keypoints': data['keypoints1'],
        })

        pred = {}
        pred.update({k + "0": v for k, v in pred0.items()})
        pred.update({k + "1": v for k, v in pred1.items()})
        
        loss, log_dict = self.raw_model.loss(data, pred, self.args)
        
        loss.backward()
        # batch accumulation
        if (self.global_batch % self.args.batch_accumulation == 0 and self.global_batch > 0) or (self.args.batch_accumulation == 1):
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        if self.gpu_id == 0  or self.gpu_id == "cpu":
            self.add_scalars("0_train/loss", loss)
            for key, log_val in log_dict.items():
                self.add_scalars(key, log_val)
                
        return loss.item(), pred

    def _run_epoch(self, epoch):
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {self.args.batch_size} | Steps: {len(self.train_data)}")
        if not self.args.local:
            self.train_data.sampler.set_epoch(epoch)
        self.model.train(True)
        self.optimizer.zero_grad()
        pbar = tqdm(self.train_data, disable=(self.gpu_id != 0))
        for idx, data in enumerate(pbar):
            data = batch_to_device(data, self.gpu_id)
            loss, pred = self._run_batch(data)

            if self.gpu_id == 0  or self.gpu_id == "cpu":
                pbar.set_description(f"Epoch {epoch} | Step {idx} | Loss: {loss:.4f}")
                
                if self.global_batch % self.args.log_every == 0:
                    self.log_all()
                if self.global_batch % self.args.plot_every == 0:
                    self.plot_qualitatives(data, pred)
                if self.global_batch % self.args.save_every == 0:
                    self._save_checkpoint()
                
            self.global_iter += (self.world_size * self.args.batch_size)
            self.global_batch += 1
        
        self.global_epoch += 1
        
    def _save_checkpoint(self):
        PATH = f"{self.model_folder}/checkpoint_{self.global_epoch}_{self.global_iter}.pt"
        ckp = {
            "state_dict": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.global_epoch,
            "iter": self.global_iter,
            "batch": self.global_batch,
        }
        torch.save(ckp, PATH)        
        print(f"Epoch {self.global_epoch} | Training checkpoint saved at {PATH}")

    def _load_checkpoint(self, path):
        ckp = torch.load(path, map_location=f"cuda:{self.gpu_id}")
        self.raw_model.load_state_dict(ckp["state_dict"])
        self.optimizer.load_state_dict(ckp["optimizer"])
        self.global_epoch = ckp["epoch"]
        self.global_iter = ckp["iter"]
        self.global_batch = ckp["batch"]
        print(f"Checkpoint loaded from {path} | Epoch {self.global_epoch} | Iter {self.global_iter}")

    def train(self, max_epochs: int):
        for epoch in range(self.global_epoch, max_epochs):
            self._run_epoch(epoch)
            if not self.args.debug and self.gpu_id == 0  or self.gpu_id == "cpu" and epoch % self.save_every == 0:
                self._save_checkpoint()

def infer_dims(name):
    if 'aliked' in name:
        return 128
    if 'alike-n' in name:
        return 128
    if 'vits14' in name:
        return 384
    if 'xfeat' in name:
        return 64
    if 'vitb14' in name:
        return 768
    if 'vitl14' in name:
        return 1024
    if 'vitg14' in name:
        return 1536
    if 'superpoint' in name:
        return 256
    if 'dedode-' in name:
        return 256
    if 'disk-' in name:
        return 128
    if 'delf-' in name:
        return 40 
    if 'relf-' in name:
        return 1024
    raise ValueError("Unknown model " + name)

def infer_model(name):
    if 'aliked' in name:
        return 'aliked-n'
    if 'alike-n' in name:
        return 'alike-n'
    if 'vits14' in name:
        return 'dinov2_vits14'
    if 'xfeat' in name:
        return 'xfeat'
    if 'vitb14' in name:
        return 'dinov2_vitb14'
    if 'vitl14' in name:
        return 'dinov2_vitl14'
    if 'vitg14' in name:
        return 'dinov2_vitg14'
    if 'superpoint' in name:
        return 'superpoint'
    if 'dedode-G' in name:
        return 'dedode-G'
    if 'dedode-B' in name:
        return 'dedode-B'
    if 'disk-' in name:
        return 'disk'
    if 'delf-' in name:
        return 'delf'
    if 'relf-' in name:
        return 'relf'
    raise ValueError("Unknown model " + name )

def infer_keypoints(name):
    if '2048' in name:
        return 2048
    if '1024' in name:
        return 1024
    if '512' in name:
        return 512
    if '256' in name:
        return 256
    if '4096' in name:
        return 4096

def load_train_objs(args, rank):
    world_size = torch.cuda.device_count()
    print(f"[GPU{rank}] Loading dataset")
    if args.debug:
        train_set = ScannetH5(args.data, 'train', max_samples=args.batch_size*world_size, features_cache=args.extractor_cache, dino_cache=args.dino_cache)
    else:
        train_set = ScannetH5(args.data, 'train', features_cache=args.extractor_cache, dino_cache=args.dino_cache)
    print(f"[GPU{rank}] Dataset loaded")

    print(f"[GPU{rank}] Loading model")
    from reasoning.features.desc_reasoning import ReasoningBase

    extractor_features_dim = infer_dims(args.extractor_cache)
    semantic_features_dim = infer_dims(args.dino_cache)
    
    config = {
        "dino":{
            "weights": infer_model(args.dino_cache),
            "allow_resize": True,
        },
        
        "extractor":{
            "model_name": infer_model(args.extractor_cache),
            "max_num_keypoints": infer_keypoints(args.extractor_cache),
            "detection_threshold": 0.0,
            "force_num_keypoints": True,
            "pretrained": True,
            "nms_radius": 2,
        },
        "reasoning":{
            "extractor_features_dim": extractor_features_dim,
            "semantic_features_dim": semantic_features_dim,
            "features_dim": 256,
            "n_attention_layers": args.layers,
            "num_heads": 1,
            "attention": "full", # full, linear
        },
        "semantic_interpolation_mode": "bicubic",
        "activate_timers": False,
        "reasoning_triplet": False,
        "attention_progression": "alternating" ,#"alternating", # alternating, semantic, visual
        "deep_supervision": True,
        "semantic_conditioning": True,
        "fix_dino_size": -1,
    }
    
    if args.resume:
        # get the config from the checkpoint folder
        folder = os.path.dirname(args.resume)
        with open(f"{folder}/model_config.yaml", "r") as f:
            config = OmegaConf.load(f)
    
    model = ReasoningBase(config)

    print(f"[GPU{rank}] Model loaded")

    print(f"[GPU{rank}] Loading optimizer")
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    print(f"[GPU{rank}] Optimizer loaded")
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, args):
    if args.local:
        sampler = None
    else:
        sampler = DistributedSampler(dataset, shuffle=True)
    
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        # pin_memory=True,
        shuffle=(sampler is None) and not args.debug,
        sampler=sampler,
        num_workers=8,
    )


def main(rank: int, world_size: int, args):
    save_every = args.save_every
    total_epochs = args.total_epochs
    batch_size = args.batch_size
    snapshot_path = "snapshot.pth"
    print(f"[GPU{rank}] Starting training job")
    if not args.local:
        ddp_setup(rank, world_size)
    
    dataset, model, optimizer = load_train_objs(args, rank)
    print(f"[GPU{rank}] Training objects loaded")
    
    if not args.local:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print(f"[GPU{rank}] SyncBatchNorm converted")    
    
    train_data = prepare_dataloader(dataset, args)
    print(f"[GPU{rank}] Dataloader prepared")
    trainer = Trainer(model, train_data, optimizer, rank, save_every, args)
    print(f"[GPU{rank}] Trainer created")
    trainer.train(total_epochs)
    print(f"[GPU{rank}] Training job finished")
    if not args.local:
        destroy_process_group()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    
    
    dino_models = [
        'dinov2_vits14',
        'dinov2_vitb14',
        'dinov2_vitl14',
        'dinov2_vitg14',
    ]

    # general arguments
    parser.add_argument('--debug', action='store_true', help='Debug mode', default=False)
    parser.add_argument('--log_every', default=10, type=int, help='Log every N steps')
    parser.add_argument('--plot_every', default=100, type=int, help='Log every N steps')
    parser.add_argument('--save_every', type=int, help='How often to save a snapshot', default=1_000)
    parser.add_argument('--comment', '-C', type=str, help='Comment for Tensorboard', default="")

    # dataset arguments
    parser.add_argument('--data', default='h5_scannet', type=str, help='Dataset location')
    parser.add_argument('--batch_size', default=2, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--total_epochs', type=int, help='Total epochs to train the model', default=1000)
    
    # trainig arguments
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_final", type=float, default=1e-4)
    parser.add_argument("--decay_epochs", type=int, default=40)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--local", action="store_true", default=False, help="Run in local mode (no DDP) (default: False)")
    parser.add_argument("--batch_accumulation", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Resume training from a checkpoint")

    # reasoning arguments
    parser.add_argument("--dino_model", type=str, default="dinov2_vits14", help="Dino model", choices=dino_models)
    parser.add_argument("--dino_cache", type=str, default="dino-scannet-dinov2_vits14", help="Dino cache location")
    parser.add_argument("--extractor_cache", type=str, default="xfeat-scannet-n2048", help="Extractor cache location")
    parser.add_argument("--layers", type=int, default=5)
    
    # model selection
    parser.add_argument("--model", type=str, default="reasoning", choices=["reasoning", "not_a_matcher"])

    args = parser.parse_args()
    # transform args to omegaconf
    args = OmegaConf.create(vars(args))
    
    world_size = torch.cuda.device_count()
    print(f"World size: {world_size}")

    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        PORT = next_free_port()
        os.environ["MASTER_PORT"] = str(PORT)
    
    if args.local:
        main(0, 1, args)
    else:
        mp.spawn(main, args=(world_size, args), nprocs=world_size)
