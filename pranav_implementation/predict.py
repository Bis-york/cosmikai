import torch
import json
import numpy as np
from torch import nn
import lightkurve as lk


# Same CNN structure used in training
class SmallCNN(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # add channel dim
        f = self.feature(x)
        return self.head(f).squeeze(1)
    
    
# Load trained weights
model_path = "models/trained_exoplanet_detector.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Loading model from {model_path} on {device}...")

# initialize model and load weights
model = SmallCNN(512).to(device)
state_dict = torch.load(model_path, map_location=device)

# some files are saved as full dicts; handle both cases
if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    model.load_state_dict(state_dict["model_state_dict"])
else:
    model.load_state_dict(state_dict)

model.eval()
print("[INFO] Model loaded successfully!")


# Load config file
config_path = "example_lightkurve_config.json"
with open(config_path, "r") as f:
    config = json.load(f)

target = config["target_name"]
mission = config.get("mission", "Kepler")
nbins = config.get("nbins", 512)

print(f"[INFO] Target: {target} | Mission: {mission}")

# Fetch and clean lightcurve
print("[INFO] Downloading lightcurve...")
sr = lk.search_lightcurve(target, mission=mission)
lc = sr.download_all().stitch().remove_nans()
print(f"[INFO] Downloaded {len(lc.flux)} points")

# Flatten & normalize (same steps as training)
lc = lc.flatten(window_length=401, polyorder=2)
f = lc.flux.value.astype(np.float32)
t = lc.time.value.astype(np.float32)
f = f / np.nanmedian(f)
f = (f - np.nanmedian(f)) / (np.nanstd(f) + 1e-6)

print("[INFO] Lightcurve ready for analysis!")

