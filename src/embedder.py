from abc import ABC, abstractmethod
from PIL import Image


class BaseEmbedder(ABC):
    @abstractmethod
    def get_image_embedding(self, image: Image.Image) -> list[float]: ...

    @abstractmethod
    def get_text_embedding(self, text: str) -> list[float]: ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int: ...

    @property
    @abstractmethod
    def model_key(self) -> str: ...


def get_embedder(model_key: str) -> BaseEmbedder:
    from .clip_embedder import CLIPEmbedder
    from .siglip_embedder import SiglipEmbedder

    registry: dict[str, type[BaseEmbedder]] = {
        "clip": CLIPEmbedder,
        "siglip": SiglipEmbedder,
    }
    if model_key not in registry:
        raise ValueError(f"Unknown model: '{model_key}'. Choose from {list(registry)}")
    return registry[model_key]()
