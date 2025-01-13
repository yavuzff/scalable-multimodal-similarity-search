"""
This handles the generation of embeddings for the texts.
We provide an abstract class for the embedding generation, which extended by various
embedding generation methods such as OpenAI and Sentence Transformers.
"""
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
import torch
from openai import OpenAI
import numpy as np
import logging


class TextEmbeddingGenerator(ABC):
    """
    Abstract class for generating embeddings.
    """

    @abstractmethod
    def generate_text_embeddings(self, texts: list[str], normalize_embeddings: bool = False,
                                 batch_size: int = 128) -> np.ndarray:
        """
        Generate embeddings given a list of string
        """
        pass


class SentenceTransformerEmbeddingGenerator(TextEmbeddingGenerator):
    """
    Embedding generator using HuggingFace SentenceTransformer model
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading model {model_name}...")
        self.DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(model_name, device=self.DEVICE_NAME)
        self.logger.info(f"Using device {self.DEVICE_NAME}...")
        self.logger.info(f"Loaded {model_name}")

    def generate_text_embeddings(self, texts: list[str], normalize_embeddings: bool = False,
                                 batch_size: int = 128) -> np.ndarray:
        self.logger.info(f"Generating {len(texts)} embeddings with batch size {batch_size}...")
        embeddings = self.model.encode(texts, normalize_embeddings=normalize_embeddings, precision="float32",
                                       batch_size=batch_size)
        self.logger.info("Generated embeddings!")
        return embeddings


class OpenAIEmbeddingGenerator(TextEmbeddingGenerator):
    """
    Embedding generator using OpenAI API
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI()
        self.model = model_name

    def generate_text_embeddings(self, texts: list[str], normalize_embeddings: bool = True,
                                 batch_size: int = 1) -> np.ndarray:
        if not normalize_embeddings:
            raise ValueError("OpenAI embeddings are always normalized to 1.")
        if batch_size != 1:
            raise ValueError("OpenAI API does not support batch_size input.")

        self.logger.info(f"Generating {len(texts)} embeddings...")
        response = self.client.embeddings.create(input=texts, model=self.model)
        # note that OpenAI embeddings are normalised to 1 already
        embeddings = np.array([embedding.embedding for embedding in response.data])
        self.logger.info("Done")
        return embeddings
