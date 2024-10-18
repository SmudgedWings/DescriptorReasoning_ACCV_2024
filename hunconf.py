dependencies = ['torch', 'torchvision', 'easy-local-features', 'omegaconf', 'os', 'zipfile']
import os
import torch

from reasoning.features.desc_reasoning import load_reasoning_from_checkpoint, Reasoning
weights_dict = {
    'xfeat': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat.zip",
    'superpoint': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/superpoint.zip",
    'alike': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/alike.zip",
    'aliked': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/aliked.zip",
    'dedode_B': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/dedode_B.zip",
    'dedode_G': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/dedode_G.zip",
    'xfeat-12_layers-dino_G': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-12_layers-dino_G.zip",
    'xfeat-12_layers': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-12_layers.zip",
    'xfeat-3_layers': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-3_layers.zip",
    'xfeat-7_layers': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-7_layers.zip",
    'xfeat-9_layers': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-9_layers.zip",
    'xfeat-dino-G': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-dino-G.zip",
    'xfeat-dino_B': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-dino_B.zip",
    'xfeat-dino_L': "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat-dino_L.zip"
}

def reasoning(pretrained='xfeat', **kwargs):
    if pretrained not in weights_dict:
        raise ValueError(f"Pretrained model {pretrained} not found in the dictionary. Available models: {', '.join(weights_dict.keys())}")
    
    # get torch hub cache dir
    cache_dir = torch.hub.get_dir() + "/reasoning_accv/"
    os.makedirs(cache_dir, exist_ok=True)

    # download and extract weights
    if not os.path.exists(cache_dir + pretrained + ".zip"):
        zip_weights = weights_dict[pretrained]
        torch.hub.download_url_to_file(zip_weights, dst=cache_dir + pretrained + ".zip", progress=True)

    # unzip
    if not os.path.exists(cache_dir + pretrained + "/"):
        import zipfile
        with zipfile.ZipFile(cache_dir + pretrained + ".zip", 'r') as zip_ref:
            zip_ref.extractall(cache_dir)

    pretrained_path = cache_dir + pretrained + "/"
    
    # Load the model with just the reasoning part
    semantic_reasoning = load_reasoning_from_checkpoint(pretrained_path)

    # Load the reasoning model here to use it together with the base model
    reasoning_model = Reasoning(semantic_reasoning['model'])
    reasoning_model.eval()
    return reasoning_model