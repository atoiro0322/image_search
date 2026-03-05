import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import DEVICE, MODEL_NAME


class CLIPEmbedder:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()

    def get_image_embedding(self, image: Image.Image) -> list[float]:
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            emb = self.model.get_image_features(**inputs).pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()[0].tolist()

    def get_text_embedding(self, text: str) -> list[float]:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            emb = self.model.get_text_features(**inputs).pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().float().numpy()[0].tolist()
