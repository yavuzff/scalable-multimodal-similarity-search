"""
This handles the generation of embeddings for the images
We provide an abstract class for the embedding generation, which extended by HuggingFace pipelines
"""
from abc import ABC, abstractmethod
import numpy as np
import logging
import torch
import os
from transformers import pipeline  # type: ignore
from PIL import Image


class ImageEmbeddingGenerator(ABC):
    """
    Abstract class for generating embeddings.
    """

    @abstractmethod
    def generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool) -> np.ndarray:
        """
        Generate embeddings given a list of paths for images - up to 1024 images
        """
        pass

    @abstractmethod
    def batch_generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool) -> np.ndarray:
        """
        Generate embeddings given a list of paths for images - batch process by 1024 images at a time
        """
        pass

    @staticmethod
    def verify_images_exist(image_paths: list[str]):
        """
        Verify that images at the given paths exist.
        """
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} does not exist")


class HFImageEmbeddingGenerator(ImageEmbeddingGenerator):
    """
    Embedding generator using HuggingFace ConvNextModel
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k"):
        self.logger = logging.getLogger(__name__)
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading image-feature-extraction pipeline for {model_name}...")
        self.pipe = pipeline(task="image-feature-extraction", model=model_name,
                             device=self.DEVICE, pool=True)
        self.logger.info(f"Loaded {model_name}")

    def generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool) -> np.ndarray:
        ImageEmbeddingGenerator.verify_images_exist(image_paths)

        images = [Image.open(image_path) for image_path in image_paths]
        embeddings = self.__generate_image_embeddings(images)

        if normalize_embeddings:
            embeddings = self.__normalize_embeddings(embeddings)

        return embeddings

    def __generate_image_embeddings(self, images: list) -> np.ndarray:
        self.logger.info(f"Generating {len(images)} embeddings...")
        outputs = self.pipe(images)
        embeddings = np.squeeze(outputs, axis=1)
        self.logger.info(f"Generated {len(images)} embeddings!")
        return embeddings

    @staticmethod
    def __normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def batch_generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool) -> np.ndarray:
        ImageEmbeddingGenerator.verify_images_exist(image_paths)

        self.logger.info(f"Generating {len(image_paths)} embeddings with batches...")

        embeddings: list[list[float]] = []
        batch_size = 1000
        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i:i + batch_size]
            batch_embeddings = self.__generate_image_embeddings(batch_image_paths)
            embeddings.extend(batch_embeddings)

        np_embeddings = np.array(embeddings)

        if normalize_embeddings:
            np_embeddings = self.__normalize_embeddings(np_embeddings)

        self.logger.info(f"Finished generation of {len(image_paths)} embeddings through batches!")

        return np_embeddings
