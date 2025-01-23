"""
This handles the generation of embeddings for the images
We provide an abstract class for the embedding generation, which extended by HuggingFace pipelines
"""
from abc import ABC, abstractmethod
import numpy as np
import logging
import torch
import os
from transformers import pipeline
from PIL import Image
from tqdm import tqdm


class ImageEmbeddingGenerator(ABC):
    """
    Abstract class for generating embeddings.
    """

    @abstractmethod
    def generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool = False) -> np.ndarray:
        """
        Generate embeddings given a list of paths for images - up to 1024 images
        """
        pass

    @abstractmethod
    def batch_generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool = False, batch_size: int = 128) -> np.ndarray:
        """
        Generate embeddings given a list of paths for images - batch process by batch_size images at a time
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

    @staticmethod
    def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """
        Normalize the list of embeddings.
        """
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)


class HFImageEmbeddingGenerator(ImageEmbeddingGenerator):
    """
    Embedding generator using HuggingFace ConvNextModel
    """

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k", batch_size=128):
        self.logger = logging.getLogger(__name__)
        self.batch_size = batch_size
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Loading image-feature-extraction pipeline for {model_name}...")
        self.pipe = pipeline(task="image-feature-extraction", model=model_name,
                             device=self.DEVICE, pool=True, torch_dtype=torch.float32)
        self.logger.info(f"Loaded {model_name}")

    def generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool = False) -> np.ndarray:
        self.verify_images_exist(image_paths)
        images = [Image.open(image_path) for image_path in image_paths]

        self.logger.info(f"Generating {len(images)} embeddings...")
        embeddings = self.__generate_image_embeddings(images)
        self.logger.info(f"Generated {len(images)} embeddings!")

        if normalize_embeddings:
            self.logger.info("Normalizing embeddings...")
            embeddings = self.normalize_embeddings(embeddings)

        return embeddings

    def batch_generate_image_embeddings(self, image_paths: list[str], normalize_embeddings: bool = False) -> np.ndarray:
        self.verify_images_exist(image_paths)
        self.logger.info(f"Generating {len(image_paths)} embeddings with batches of size {self.batch_size}...")

        np_embeddings = self.__generate_image_embeddings(image_paths)

        if normalize_embeddings:
            self.logger.info("Normalizing embeddings...")
            np_embeddings = self.normalize_embeddings(np_embeddings)

        self.logger.info(f"Finished generation of {len(image_paths)} embeddings through batches of size {self.batch_size}!")

        return np_embeddings

    def __generate_image_embeddings(self, images: list) -> np.ndarray:
        outputs = []

        # Process images in batches with tqdm
        for i in tqdm(range(0, len(images), self.batch_size), desc="Generating image embeddings"):
            batch = images[i:i + self.batch_size]
            outputs.extend(self.pipe(batch))
        
        outputs = np.array(outputs)
        embeddings = np.squeeze(outputs, axis=1)
        # cast embeddings to float32
        embeddings = embeddings.astype(np.float32)
        return embeddings
