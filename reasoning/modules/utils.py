import torch
import torchvision
import cv2
import numpy as np
import h5py
import torch


def load_image(path, gray=False):
    '''Loads an image from a file path and returns it as a torch tensor
    Output shape: (3, H, W) float32 tensor with values in the range [0, 1]
    '''
    image = torchvision.io.read_image(str(path)).float() / 255
    if gray:
        image = torchvision.transforms.functional.rgb_to_grayscale(image)
    return image

def load_depth(path):
    image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    image = torch.tensor(image).unsqueeze(0)
    return image

def to_cv(torch_image, convert_color=True, batch_idx=0, to_gray=False):
    '''Converts a torch tensor image to a numpy array'''
    if isinstance(torch_image, torch.Tensor):
        if len(torch_image.shape) == 2:
            torch_image = torch_image.unsqueeze(0)
        if len(torch_image.shape) == 4 and torch_image.shape[0] == 1:
            torch_image = torch_image[0]
        if len(torch_image.shape) == 4 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx]
        if len(torch_image.shape) == 3 and torch_image.shape[0] > 1:
            torch_image = torch_image[batch_idx].unsqueeze(0)
            
        if torch_image.max() > 1:
            torch_image = torch_image / torch_image.max()
        
        img = (torch_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    else:
        img = torch_image

    if convert_color:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img

def batch_to_device(batch, device):
    '''Moves a batch of tensors to a device'''
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: batch_to_device(v, device) for k, v in batch.items()}


# --- DATA IO ---

def imread_gray(path, augment_fn=None):
    image = cv2.imread(str(path), 1)
    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)

def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new

def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def fix_path_from_d2net(path):
    if not path:
        return None
    
    path = path.replace('phoenix/S6/zl548/MegaDepth_v1/', '')
    scene_id = path.split('/')[-4]
    path = path.replace(f'{scene_id}/dense0/depths', f'depth_undistorted/{scene_id}')
    return path

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    image = imread_gray(path, augment_fn)

    # resize image
    w, h = image.shape[1], image.shape[0]

    if len(resize) == 2:
        w_new, h_new = resize
    else:
        resize = resize[0]
        w_new, h_new = get_resized_wh(w, h, resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    #image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float().permute(2,0,1) / 255  # (h, w) -> (1, h, w) and normalized
    mask = torch.from_numpy(mask) if mask is not None else None

    return image, mask, scale

def resize_masks(masks, resize=None, df=None, padding=False):
    """
    Args:
        masks (torch.tensor): (b, h, w)
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
    Returns:
        masks (torch.tensor): (b, h_new, w_new)
    """
    b, h, w = masks.shape
    if len(resize) == 2:
        w_new, h_new = resize
    else:
        resize = resize[0]
        w_new, h_new = get_resized_wh(w, h, resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)

    masks = torch.nn.functional.interpolate(masks, (h_new, w_new), mode='nearest').squeeze(1)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        masks = torch.stack([pad_bottom_right(mask, pad_to, ret_mask=False) for mask in masks])
    else:
        masks = masks.unsqueeze(1)

    return masks

def read_megadepth_depth(path, pad_to=None):
    depth = np.array(h5py.File(path, 'r')['depth'])
    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    depth = torch.from_numpy(depth).float()  # (h, w)
    return depth

if __name__ == "__main__":
    from reasoning.utils import root
    
    image_path = root / "assets" / "boat2.png"
    img = load_image(image_path)
    
    assert img.shape == (3, 680, 850), "Image shape is not as expected"
    assert img.dtype == torch.float32, "Image dtype is not as expected"
    assert img.max() <= 1, "Image values are not normalized"
    assert img.min() >= 0, "Image values are not normalized"