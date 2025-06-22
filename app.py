import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np

# Generator class definition
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        return img.view(img.size(0), 1, 28, 28)

# Load generator
device = torch.device("cpu")
generator = Generator()
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

# Streamlit App UI
st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")

digit = st.selectbox("Choose a digit to generate (0–9):", list(range(10)))
if st.button("Generate Images"):
    z = torch.randn(5, 100)
    labels = torch.tensor([digit] * 5)
    with torch.no_grad():
        imgs = generator(z, labels)

    grid = make_grid(imgs, nrow=5, normalize=True)
    npimg = grid.numpy()
    st.image(np.transpose(npimg, (1, 2, 0)), caption=[f"Sample {i+1}" for i in range(5)])