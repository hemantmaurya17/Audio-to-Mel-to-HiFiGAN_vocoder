import torch
import json
from types import SimpleNamespace
from models import Generator
from scipy.io.wavfile import write
import numpy as np

# 1. Load config.json
with open("config_v1.json") as f:
    config_dict = json.load(f)
config = SimpleNamespace(**config_dict)

# 2. Load Generator model
model = Generator(config)
checkpoint = torch.load("generator_v1.pt", map_location="cpu")
model.load_state_dict(checkpoint['generator'])  # Make sure 'generator' key matches
model.eval()

# 3. Dummy MEL spectrogram (for testing)
mel = torch.randn(1, 80, 620)  # [batch, num_mels, time]

# 4. Generate Audio from MEL
with torch.no_grad():
    audio = model(mel).squeeze().cpu().numpy()

# 5. Save Audio
write("output.wav", config.sampling_rate, (audio * 32767).astype(np.int16))

print("✅ Audio Generated Successfully: output.wav")
