"""CLIPEmbedder と SiglipEmbedder のユニットテスト（モデルはモック）"""

import math
from unittest.mock import MagicMock, patch

import torch
import pytest
from PIL import Image


def _make_fake_output(dim: int):
    """pooler_output が dim 次元のゼロテンソルを返すモック出力を作る"""
    tensor = torch.zeros(1, dim)
    out = MagicMock()
    out.pooler_output = tensor
    return out


# ---- CLIPEmbedder ----

@pytest.fixture
def clip_embedder():
    with patch("src.clip_embedder.CLIPModel") as MockModel, \
         patch("src.clip_embedder.CLIPProcessor") as MockProcessor:

        instance = MockModel.from_pretrained.return_value
        instance.to.return_value = instance
        instance.get_image_features.return_value = _make_fake_output(512)
        instance.get_text_features.return_value = _make_fake_output(512)

        proc_output = MagicMock()
        proc_output.to.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        MockProcessor.from_pretrained.return_value = MagicMock(return_value=proc_output)

        from src.clip_embedder import CLIPEmbedder
        return CLIPEmbedder()


def test_clip_get_image_embedding_returns_list(clip_embedder):
    result = clip_embedder.get_image_embedding(Image.new("RGB", (224, 224)))
    assert isinstance(result, list)


def test_clip_get_image_embedding_dim(clip_embedder):
    result = clip_embedder.get_image_embedding(Image.new("RGB", (224, 224)))
    assert len(result) == 512


def test_clip_get_text_embedding_returns_list(clip_embedder):
    assert isinstance(clip_embedder.get_text_embedding("a dog"), list)


def test_clip_get_text_embedding_dim(clip_embedder):
    assert len(clip_embedder.get_text_embedding("a dog")) == 512


def test_clip_embedding_normalized():
    with patch("src.clip_embedder.CLIPModel") as MockModel, \
         patch("src.clip_embedder.CLIPProcessor") as MockProcessor:

        tensor = torch.tensor([[1.0, 2.0, 3.0] + [0.0] * 509])
        out = MagicMock()
        out.pooler_output = tensor
        instance = MockModel.from_pretrained.return_value
        instance.to.return_value = instance
        instance.get_image_features.return_value = out
        proc_output = MagicMock()
        proc_output.to.return_value = {}
        MockProcessor.from_pretrained.return_value = MagicMock(return_value=proc_output)

        from src.clip_embedder import CLIPEmbedder
        embedder = CLIPEmbedder()

    result = embedder.get_image_embedding(Image.new("RGB", (224, 224)))
    norm = math.sqrt(sum(v ** 2 for v in result))
    assert norm == pytest.approx(1.0, abs=1e-5)


def test_clip_model_key():
    from src.clip_embedder import CLIPEmbedder
    assert CLIPEmbedder.model_key == "clip"


def test_clip_embedding_dim_property():
    from src.clip_embedder import CLIPEmbedder
    assert CLIPEmbedder.embedding_dim == 512


# ---- SiglipEmbedder ----

@pytest.fixture
def siglip_embedder():
    with patch("src.siglip_embedder.SiglipModel") as MockModel, \
         patch("src.siglip_embedder.SiglipProcessor") as MockProcessor:

        instance = MockModel.from_pretrained.return_value
        instance.to.return_value = instance
        instance.get_image_features.return_value = _make_fake_output(768)
        instance.get_text_features.return_value = _make_fake_output(768)

        proc_output = MagicMock()
        proc_output.to.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        MockProcessor.from_pretrained.return_value = MagicMock(return_value=proc_output)

        from src.siglip_embedder import SiglipEmbedder
        return SiglipEmbedder()


def test_siglip_get_image_embedding_returns_list(siglip_embedder):
    result = siglip_embedder.get_image_embedding(Image.new("RGB", (224, 224)))
    assert isinstance(result, list)


def test_siglip_get_image_embedding_dim(siglip_embedder):
    result = siglip_embedder.get_image_embedding(Image.new("RGB", (224, 224)))
    assert len(result) == 768


def test_siglip_get_text_embedding_returns_list(siglip_embedder):
    assert isinstance(siglip_embedder.get_text_embedding("a dog"), list)


def test_siglip_get_text_embedding_dim(siglip_embedder):
    assert len(siglip_embedder.get_text_embedding("a dog")) == 768


def test_siglip_model_key():
    from src.siglip_embedder import SiglipEmbedder
    assert SiglipEmbedder.model_key == "siglip"


def test_siglip_embedding_dim_property():
    from src.siglip_embedder import SiglipEmbedder
    assert SiglipEmbedder.embedding_dim == 768


# ---- get_embedder ----

def test_get_embedder_clip():
    with patch("src.clip_embedder.CLIPModel"), patch("src.clip_embedder.CLIPProcessor"):
        from src.embedder import get_embedder
        from src.clip_embedder import CLIPEmbedder

        e = get_embedder("clip")
        assert isinstance(e, CLIPEmbedder)


def test_get_embedder_unknown_raises():
    from src.embedder import get_embedder
    with pytest.raises(ValueError, match="Unknown model"):
        get_embedder("unknown_model")
