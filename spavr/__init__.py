__all__ = ["CJepaBackbone", "QuantileHead", "pinball_loss", "RewardRiskModel"]


def __getattr__(name):
    if name == "CJepaBackbone":
        from .backbone import CJepaBackbone

        return CJepaBackbone
    if name in {"QuantileHead", "pinball_loss"}:
        from .heads import QuantileHead, pinball_loss

        return {"QuantileHead": QuantileHead, "pinball_loss": pinball_loss}[name]
    if name == "RewardRiskModel":
        from .model import RewardRiskModel

        return RewardRiskModel
    raise AttributeError(name)
