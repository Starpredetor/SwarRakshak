"""L1: the trained anti-spoof layer.

Wraps a HuggingFace audio-classification checkpoint (wav2vec2 / AASIST-style)
that was fine-tuned to separate genuine speech from vocoder output. TTS
vocoders reconstruct the spectrum statistically, which leaves the vocal-tract
resonances over-smoothed and slightly mistimed -- artifacts that survive
translation across languages, which is why this layer carries most of the
fused weight.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.detector.base import Detector
from backend.models import LayerScore

logger = logging.getLogger(__name__)

# Substrings that mark the synthetic class. Checkpoints are inconsistent:
# "fake", "spoof", "spoofed", "synthetic", "LA" (ASVspoof logical access) all
# appear in the wild.
_SPOOF_HINTS = ("fake", "spoof", "synth", "clone", "deepfake", "generated")
_BONAFIDE_HINTS = ("real", "bona", "genuine", "human", "authentic")


class HFSpoofDetector(Detector):
    name = "l1_spoof"

    def __init__(self, model_id: str, device: str, spoof_label: str = "fake") -> None:
        self.model_id = model_id
        self.device = device
        self.spoof_label = spoof_label
        self._model = None
        self._extractor = None
        self._spoof_index: int | None = None

    def load(self) -> None:
        """Pull the checkpoint, move it to `device`, and warm up.

        Warmup matters: the first CUDA forward pass costs far more than the
        rest -- kernel autotuning, lazy module init -- and without it the very
        first window of the demo blows the 300 ms budget on camera.
        """
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        from backend.config import get_config, get_settings

        cache_dir = str(get_settings().model_dir)
        logger.info("loading %s onto %s", self.model_id, self.device)

        self._extractor = AutoFeatureExtractor.from_pretrained(
            self.model_id, cache_dir=cache_dir
        )
        self._model = AutoModelForAudioClassification.from_pretrained(
            self.model_id, cache_dir=cache_dir
        ).to(self.device).eval()

        self._spoof_index = self._resolve_spoof_index()
        logger.info(
            "%s: id2label=%s -> spoof index %d",
            self.model_id, self._model.config.id2label, self._spoof_index,
        )

        sr = get_config()["audio"]["sample_rate"]
        window = get_config()["audio"]["window_seconds"]
        iters = get_config()["detector"].get("warmup_iters", 3)
        dummy = np.zeros(int(sr * window), dtype=np.float32)
        with torch.no_grad():
            for _ in range(iters):
                self._forward(dummy, sr)

    def _resolve_spoof_index(self) -> int:
        """Map the synthetic class onto the checkpoint's id2label.

        Checkpoints disagree about label order -- some put 'fake' at index 0,
        some at 1. Getting this backwards inverts the entire demo while still
        looking like a working detector, so it is resolved explicitly at load
        time and never assumed.

        Raises if the labels cannot be read confidently. An unresolvable
        mapping is a refusal to start, not a coin flip.
        """
        id2label = getattr(self._model.config, "id2label", None)
        if not id2label:
            raise RuntimeError(f"{self.model_id}: checkpoint exposes no id2label")

        labels = {int(i): str(name).strip().lower() for i, name in id2label.items()}

        # 1. Exact match on the configured label.
        want = self.spoof_label.strip().lower()
        for idx, name in labels.items():
            if name == want:
                return idx

        # 2. Substring hint for the synthetic class.
        hits = [i for i, n in labels.items() if any(h in n for h in _SPOOF_HINTS)]
        if len(hits) == 1:
            return hits[0]

        # 3. Two labels and one is clearly the genuine class: take the other.
        if len(labels) == 2:
            bona = [i for i, n in labels.items() if any(h in n for h in _BONAFIDE_HINTS)]
            if len(bona) == 1:
                return next(i for i in labels if i != bona[0])

        raise RuntimeError(
            f"{self.model_id}: cannot tell which label means synthetic from "
            f"{labels!r}. Set detector.spoof_label in config.yaml to one of them."
        )

    def _forward(self, wav: np.ndarray, sample_rate: int):
        """One forward pass. Returns the softmax over classes as a 1-D tensor."""
        import torch

        inputs = self._extractor(
            wav, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self._model(**inputs).logits
        return torch.softmax(logits, dim=-1)[0]

    def score(self, wav: np.ndarray, sample_rate: int) -> LayerScore:
        import torch

        try:
            with torch.no_grad():
                probs = self._forward(wav, sample_rate)
            p_spoof = float(probs[self._spoof_index].item())
        except Exception:
            # Never raise into the pipeline. A layer that cannot answer
            # abstains; one bad window must not end a live call.
            logger.exception("%s failed on a window; abstaining", self.name)
            return LayerScore(layer=self.name, score=0.0, confidence=0.0, abstain=True)

        return LayerScore(
            layer=self.name,
            score=p_spoof,
            # Distance from the decision boundary, rescaled to [0, 1]. A
            # checkpoint sitting at 0.5 is telling you it cannot separate this
            # window, and the UI should show that rather than a bare number.
            confidence=abs(p_spoof - 0.5) * 2.0,
            detail={"p_spoof": p_spoof},
        )

    def close(self) -> None:
        import torch

        self._model = None
        self._extractor = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
