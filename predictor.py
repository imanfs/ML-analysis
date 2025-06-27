from pathlib import Path
import json
import faiss
import faiss.contrib.torch_utils  # noqa: F401
import torch
import numpy as np

from utils import log


class DistancePredictor:
    def __init__(
        self,
        faiss_index_path: Path,
        class_names_path: Path,
        use_gpu: bool,
        k_nearest=50,
        supervectors_path: Path = None,
        filtered_classes_path: Path = None,
    ):
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        self.use_gpu = use_gpu

        if use_gpu:
            log("setting faiss to use gpu")
            self.faiss_index = faiss.index_cpu_to_all_gpus(self.faiss_index)
        if supervectors_path is not None:
            self.supervectors = torch.load(supervectors_path, map_location="cpu")

        with open(class_names_path, "r") as f:
            superclasses = json.load(f)
            self.superclasses = np.array(superclasses)

        if filtered_classes_path:
            with open(filtered_classes_path, "r") as f:
                filtered_class_names = json.load(f)
            cls_to_vec = {
                cls: vec
                for cls, vec in zip(self.superclasses, self.supervectors)
                if cls in filtered_class_names
            }
            self.superclasses = np.array(list(cls_to_vec.keys()))
            self.supervectors = torch.stack(
                [cls_to_vec[cls] for cls in self.superclasses]
            )

        self.k = k_nearest

    def predict(self, embedded_vectors: torch.Tensor):
        if embedded_vectors.is_mps:
            embedded_vectors = embedded_vectors.cpu()

        distances, indices = self.faiss_index.search(embedded_vectors, k=self.k)

        if self.use_gpu:
            indices = indices.cpu()

        labels = self.superclasses[indices]

        return distances, labels

    def predict_pt(self, embedded_vectors: torch.Tensor):
        if embedded_vectors.is_mps:
            embedded_vectors = embedded_vectors.cpu()
        if self.supervectors.is_mps:
            supervectors = self.supervectors.cpu()
        else:
            supervectors = self.supervectors

        # Compute distances: shape = (num_queries, num_supervecs)
        distances = torch.cdist(
            embedded_vectors, supervectors, p=2
        )  # Use p=2 for Euclidean

        # Get top-k indices (use dim instead of axis)
        indices = torch.argsort(distances, dim=1)[:, : self.k]

        # Select distances and corresponding labels
        topk_distances = torch.gather(distances, dim=1, index=indices)
        topk_labels = self.superclasses[indices.cpu().numpy()]  # numpy indexing

        return topk_distances, topk_labels
