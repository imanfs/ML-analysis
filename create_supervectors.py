import json

import os
from pathlib import Path

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torchvision import transforms

import numpy as np
from tqdm import tqdm
from embedder import Embedder
from PIL import Image


def create_vectors(model_root, crops_dir, device, output_dir=None):
    embedder = Embedder(
        model_root / "trunk_weights.pth",
        model_root / "embedder_weights.pth",
        device=device,
    )

    batch_size = 1500  # ← tweak to taste

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize(
        (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
    )

    image_paths = sorted(Path(crops_dir).rglob("*.jpg"))

    all_vecs = []  # will hold the per-batch outputs
    classes = []  # will hold all classes
    for start in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[start : start + batch_size]

        batch_imgs = [
            resize(to_tensor(Image.open(p).convert("RGB"))) for p in batch_paths
        ]
        batch_tensor = torch.stack(batch_imgs).to(device)

        batch_vecs = embedder.embed(batch_tensor)
        all_vecs.append(batch_vecs)
        batch_classes = [path.stem for path in batch_paths]
        classes.extend(batch_classes)
    vectors = torch.cat(all_vecs, dim=0)
    classes = [img.stem for img in sorted((crops_dir).rglob("*.jpg"))]
    if output_dir:
        torch.save(vectors, output_dir / "vectors.pt")
        json.dump(classes, open(output_dir / "classes.json", "w"))

    return vectors, classes


def create_supervectors(vectors, classes, output_dir=None):
    unique_classes, class_indices = np.unique(
        ["_".join(cls.split("_")[:-1]) for cls in classes], return_inverse=True
    )
    num_of_classes = class_indices[-1] + 1
    # num_uuids = uuid_indices[-1] + 1
    supervectors = torch.empty((num_of_classes, 256))
    for class_i in tqdm(range(num_of_classes)):
        vectors_i = vectors[class_indices == class_i]
        supervectors[class_i, :] = torch.mean(vectors_i, axis=0)

    ## need to normalize the supervectors
    supervectors /= torch.linalg.norm(supervectors, axis=1)[:, torch.newaxis]
    superclasses = unique_classes
    if output_dir:
        torch.save(
            supervectors,
            output_dir / "supervectors.pt",
        )
        with open(output_dir / "superclasses.json", "w") as f:
            json.dump(superclasses.tolist(), f)

    return supervectors, superclasses


def main():

    ROOT_DIR = Path("/Users/iman/345-data/ml-datasets/ccbf")

    DATASET = "ccbf-det-fronts-20250709"
    output_dir = ROOT_DIR / "vectors" / DATASET
    crops_dir = ROOT_DIR / "recognition" / DATASET

    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps")
    model_root = Path("./model")

    vectors, classes = create_vectors(model_root, crops_dir, device, output_dir)
    # can load in from pt also
    vectors = torch.load(output_dir / "vectors.pt")
    classes = json.load(open(output_dir / "classes.json"))
    supervectors, superclasses = create_supervectors(vectors, classes, output_dir)

    print(f"Created {len(supervectors)} supervectors from {len(vectors)} vectors.")


if __name__ == "__main__":
    main()
