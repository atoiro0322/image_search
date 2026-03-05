"""CLIPEmbedder のユニットテスト（CLIPモデルはモック）"""

from unittest.mock import MagicMock, patch

import torch
import pytest
from PIL import Image


def _make_fake_output(dim: int = 512):
    """pooler_output が dim 次元のゼロテンソルを返すモック出力を作る"""
    tensor = torch.zeros(1, dim)
    out = MagicMock()
    out.pooler_output = tensor
    return out


@pytest.fixture
def embedder():
    with patch("src.embedder.CLIPModel") as MockModel, \
         patch("src.embedder.CLIPProcessor") as MockProcessor:

        instance = MockModel.from_pretrained.return_value
        instance.to.return_value = instance
        instance.get_image_features.return_value = _make_fake_output()
        instance.get_text_features.return_value = _make_fake_output()

        proc_output = MagicMock()
        proc_output.to.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}
        MockProcessor.from_pretrained.return_value = MagicMock(return_value=proc_output)

        from src.embedder import CLIPEmbedder
        return CLIPEmbedder()


def test_get_image_embedding_returns_list(embedder):
    img = Image.new("RGB", (224, 224))
    result = embedder.get_image_embedding(img)
    assert isinstance(result, list)


def test_get_image_embedding_dim(embedder):
    img = Image.new("RGB", (224, 224))
    result = embedder.get_image_embedding(img)
    assert len(result) == 512


def test_get_text_embedding_returns_list(embedder):
    result = embedder.get_text_embedding("a dog running outside")
    assert isinstance(result, list)


def test_get_text_embedding_dim(embedder):
    result = embedder.get_text_embedding("a dog running outside")
    assert len(result) == 512


def test_get_image_embedding_normalized():
    """L2正規化後のノルムは1になること（非ゼロのテンソルで検証）"""
    with patch("src.embedder.CLIPModel") as MockModel, \
         patch("src.embedder.CLIPProcessor") as MockProcessor:

        tensor = torch.tensor([[1.0, 2.0, 3.0] + [0.0] * 509])  # shape (1, 512)
        out = MagicMock()
        out.pooler_output = tensor

        instance = MockModel.from_pretrained.return_value
        instance.to.return_value = instance
        instance.get_image_features.return_value = out

        proc_output = MagicMock()
        proc_output.to.return_value = {}
        MockProcessor.from_pretrained.return_value = MagicMock(return_value=proc_output)

        from src.embedder import CLIPEmbedder
        embedder = CLIPEmbedder()

    import math
    img = Image.new("RGB", (224, 224))
    result = embedder.get_image_embedding(img)
    norm = math.sqrt(sum(v ** 2 for v in result))
    assert norm == pytest.approx(1.0, abs=1e-5)
