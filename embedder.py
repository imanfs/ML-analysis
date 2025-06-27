import timm
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path


class L2NormalizationLayer(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2NormalizationLayer, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return nn.functional.normalize(x, p=2, dim=self.dim, eps=self.eps)


def get_trunk_embedder(
    device, embedding_dim=256, trunk_weights=None, embedder_weights=None
):
    vit = timm.create_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        pretrained=False,
        cache_dir="./cache/",
        num_classes=0,
    ).to(device)
    vit.load_state_dict(torch.load(trunk_weights, map_location=device))
    linear = nn.Sequential(
        nn.Linear(vit.num_features, embedding_dim, device=device),
        L2NormalizationLayer(),
    ).to(device)
    linear.load_state_dict(torch.load(embedder_weights, map_location=device))
    return vit, linear


def get_transforms():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


class EmbeddingModel(nn.Module):
    def __init__(self, trunk, embedder):
        super().__init__()
        self.trunk = trunk
        self.embedder = embedder

    def forward(self, x):
        return self.embedder(self.trunk(x))


class Embedder:
    def __init__(
        self, trunk_weights: Path, embedder_weights: Path, device: torch.device
    ):
        self.device = device

        trunk, embedder = get_trunk_embedder(
            device,
            embedding_dim=256,
            trunk_weights=trunk_weights,
            embedder_weights=embedder_weights,
        )
        self.model = EmbeddingModel(trunk, embedder).to(device)
        self.model.eval()
        self.transform = get_transforms()

    def embed(self, images_tensor: torch.Tensor):
        with torch.no_grad():
            images_tensor = images_tensor.to(self.device)
            embeddings = self.model(images_tensor)
        return embeddings
