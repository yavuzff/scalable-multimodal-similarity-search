"""
This handles the generation of embeddings for the input data.
We provide an abstract class for the embedding generation, which extended by various
embedding generation methods such as OpenAI and Sentence Transformers.
"""
from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import numpy as np
import logging


class EmbeddingGenerator(ABC):
    """
    Abstract class for generating embeddings.
    """

    @abstractmethod
    def generate_text_embeddings(self, texts: list[str], normalize_embeddings: bool) -> np.ndarray:
        """
        Generate embeddings given a list of string
        """
        pass


class SentenceTransformerEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator using HuggingFace SentenceTransformer model
    """

    def __init__(self, model_name: str):
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Loading model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Loaded {model_name}")

    def generate_text_embeddings(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        self.logger.info(f"Generating {len(texts)} embeddings...")
        embeddings = self.model.encode(texts, normalize_embeddings=normalize_embeddings)
        self.logger.info("Done")
        return embeddings


class OpenAIEmbeddingGenerator(EmbeddingGenerator):
    """
    Embedding generator using OpenAI API
    """

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.logger = logging.getLogger(__name__)
        self.client = OpenAI()
        self.model = model_name

    def generate_text_embeddings(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        if not normalize_embeddings:
            raise ValueError("OpenAI embeddings are always normalized to 1.")

        self.logger.info(f"Generating {len(texts)} embeddings...")
        response = self.client.embeddings.create(input=texts, model=self.model)
        # note that OpenAI embeddings are normalised to 1 already
        embeddings = np.array([embedding.embedding for embedding in response.data])
        self.logger.info("Done")
        return embeddings
