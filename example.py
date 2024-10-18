
from reasoning.features.desc_reasoning import load_reasoning_from_checkpoint, Reasoning
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# download and extract weights
if not os.path.exists("models/xfeat.zip"):
    zip_weights = "https://github.com/verlab/DescriptorReasoning_ACCV_2024/releases/download/weights/xfeat.zip" 
    os.makedirs("models", exist_ok=True)
    torch.hub.download_url_to_file(zip_weights, dst="models/xfeat.zip", progress=True)

# unzip
if not os.path.exists("models/xfeat/"):
    import zipfile
    with zipfile.ZipFile("models/xfeat.zip", 'r') as zip_ref:
        zip_ref.extractall("models/")

# Load the model with just the reasoning part
semantic_reasoning = load_reasoning_from_checkpoint('models/xfeat/')

# Load the reasoning model here to use it together with the base model
reasoning_model = Reasoning(semantic_reasoning['model'])
reasoning_model.eval()
reasoning_model.to(device)

from reasoning.datasets.utils import load_image
image0 = load_image("assets/pumpkin1.png").unsqueeze(0).to(device)
image1 = load_image("assets/pumpkin2.png").unsqueeze(0).to(device)

match_response = reasoning_model.match({
    'image0': image0,
    'image1': image1
})

mkpts0 = match_response['matches0'][0]
mkpts1 = match_response['matches1'][0]

from reasoning.modules.visualization import plot_pair, plot_matches, save

plot_pair(image0, image1)
plot_matches(mkpts0, mkpts1)
save("assets/matches.png")