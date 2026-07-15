from pathlib import Path

from omegaconf import OmegaConf
from torch import nn

from .backbone import CJepaBackbone
from .heads import QuantileHead

from src.third_party.videosaur.videosaur import models

# videosaur is config-driven: models.build must read videosaur's own config so the
# OC architecture matches the checkpoint exactly. This is cjepa's own file, not a
# hyper-param we choose -- backbone hyper-params, by contrast, are hand-filled.
_VIDEOSAUR_CFG = str(
    Path(__file__).resolve().parents[1] / "third_party" / "cjepa" / "configs" / "config_train_causal_pusht_slot.yaml"
)


class RewardRiskModel(nn.Module):
    def __init__(self, hidden_dim=256, taus=(0.1, 0.5, 0.9)):
        super().__init__()
        self.backbone = CJepaBackbone()

        cfg = OmegaConf.load(_VIDEOSAUR_CFG)
        cfg.model.load_weights = None                    # structure only; factory fills weights
        oc = models.build(cfg.model, cfg.dummy_optimizer, None, None)
        self.oc_encoder = oc.encoder                     # MapOverTime(FrameEncoder)
        self.oc_initializer = oc.initializer             # RandomInit
        self.oc_processor = oc.processor                 # ScanOverTime(LatentProcessor(SlotAttention))

        self.head = QuantileHead(self.backbone.slot_dim, self.backbone.num_objects, hidden_dim, taus)

    def forward(self, x):
        """Score one batch -> quantiles ``(B, K)``.

        ``x`` is a batch dict with ``pixels (B, T, 3, H, W)``, ``action``, and
        ``proprio``; the frames run through the OC model, then the backbone latent,
        then the head.
        """
        pixels = x["pixels"]
        features = self.oc_encoder(pixels)["features"]
        slots0 = self.oc_initializer(batch_size=pixels.shape[0])
        slots = self.oc_processor(slots0, features)["state"]
        z = self.backbone.forward_repr(slots, x["action"], x["proprio"])
        return self.head(z)


