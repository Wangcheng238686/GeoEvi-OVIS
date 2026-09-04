"""Embedding providers for the evidence graph (S3.2).

`RemoteCLIPProvider` wraps the local RemoteCLIP ViT-L-14 checkpoint via
open_clip and exposes batched image/text embedding with L2 normalisation.
Heavy calls are cached per (kind, key) so repeated graph builds over the
same image reuse features. Tests use `MockProvider` with deterministic
vectors — no GPU / no download.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


class MockProvider:
    """Deterministic provider for unit tests.

    Text embeddings are unit vectors on a per-word hash angle; image
    embeddings are unit vectors at an angle derived from the key. `affinity`
    lets tests plant exact cosines between a text term and an image key.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self._affinity: Dict[str, Dict[str, float]] = {}

    def set_affinity(self, text: str, image_key: str, cos: float) -> None:
        self._affinity.setdefault(text, {})[image_key] = cos

    def _unit(self, angle: float) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[0], v[1] = np.cos(angle), np.sin(angle)
        return v

    @staticmethod
    def _angle(text: str) -> float:
        return float(sum(ord(ch) for ch in text) % 360) * np.pi / 180.0

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([self._unit(self._angle(t)) for t in texts])

    def encode_images(self, keys: Sequence[str]) -> np.ndarray:
        return np.stack([self._unit(self._angle("img:" + k)) for k in keys])

    def encode_images_cached(self, items) -> Dict[str, np.ndarray]:
        return {k: self._unit(self._angle("img:" + k)) for k, _ in items}

    def encode_texts_cached(self, texts: Sequence) -> Dict[str, np.ndarray]:
        return {t: self._unit(self._angle(t)) for t in texts}

    def crop_text_cos(self, image_key: str, texts: Sequence[str]) -> np.ndarray:
        img = self._unit(self._angle("img:" + image_key))
        out = []
        for t in texts:
            planted = self._affinity.get(t, {}).get(image_key)
            out.append(planted if planted is not None else float(img @ self._unit(self._angle(t))))
        return np.asarray(out, dtype=np.float32)

    def crop_crop_cos(self, key_a: str, key_b: str) -> float:
        a = self._unit(self._angle("img:" + key_a))
        b = self._unit(self._angle("img:" + key_b))
        return float(a @ b)

    def crop_image_cos(self, crop_key: str) -> float:
        # crop vs its own global image: plantable, default high
        planted = self._affinity.get("__scene__", {}).get(crop_key)
        if planted is not None:
            return planted
        a = self._unit(self._angle("img:" + crop_key))
        b = self._unit(self._angle("img:__global__"))
        return float(a @ b)


class RemoteCLIPProvider:
    """RemoteCLIP ViT-L-14 via open_clip. GPU when available, else CPU."""

    def __init__(self, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        import open_clip

        ckpt = ckpt_path or "/data/wangcheng/pretrained-models/RemoteCLIP/RemoteCLIP-ViT-L-14.pt"
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=ckpt)
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model = self.model.eval().to(self.device)
        self._img_cache: Dict[str, np.ndarray] = {}
        self._txt_cache: Dict[str, np.ndarray] = {}

    @torch.no_grad()
    def _encode_images(self, pil_images) -> np.ndarray:
        t = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        f = self.model.encode_image(t)
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    @torch.no_grad()
    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        t = self.tokenizer(list(texts)).to(self.device)
        f = self.model.encode_text(t)
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

    @staticmethod
    def _batch(seq: List, n: int):
        for i in range(0, len(seq), n):
            yield seq[i:i + n]

    def encode_images_cached(self, items: Sequence) -> Dict[str, np.ndarray]:
        """items: [(key, PIL image)] -> {key: unit vec}."""
        todo = [(k, im) for k, im in items if k not in self._img_cache]
        for batch in self._batch(todo, 64):
            feats = self._encode_images([im for _, im in batch])
            for (k, _), f in zip(batch, feats):
                self._img_cache[k] = f
        return {k: self._img_cache[k] for k, _ in items}

    def encode_texts_cached(self, texts: Sequence) -> Dict[str, np.ndarray]:
        todo = [t for t in texts if t not in self._txt_cache]
        for batch in self._batch(list(todo), 256):
            feats = self._encode_texts(batch)
            for t, f in zip(batch, feats):
                self._txt_cache[t] = f
        return {t: self._txt_cache[t] for t in texts}
