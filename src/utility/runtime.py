from dataclasses import dataclass
from types import ModuleType

import torch

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    xm = None


@dataclass(frozen=True)
class Runtime:
    kind: str
    device: object
    model_init_device: object
    xla_model: ModuleType | None = None

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.kind == "xla":
            if self.xla_model is None:
                raise RuntimeError("XLA runtime was selected without torch_xla")
            self.xla_model.optimizer_step(optimizer, barrier=True)
            self.xla_model.mark_step()
            return
        optimizer.step()


def resolve_runtime(requested: str = "auto") -> Runtime:
    supported = {"auto", "xla", "cuda", "mps", "cpu"}
    if requested not in supported:
        raise ValueError(
            f"Unknown device '{requested}'. Expected one of: {', '.join(sorted(supported))}."
        )

    if requested in {"auto", "xla"} and xm is not None:
        device = xm.xla_device()
        return Runtime("xla", device, "cpu", xm)
    if requested == "xla":
        raise RuntimeError("XLA was requested, but torch_xla is not installed")

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        device = torch.device("cuda")
        return Runtime("cuda", device, device)
    if requested == "cuda":
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested in {"auto", "mps"} and mps_available:
        device = torch.device("mps")
        return Runtime("mps", device, device)
    if requested == "mps":
        raise RuntimeError("MPS was requested, but no MPS device is available")

    device = torch.device("cpu")
    return Runtime("cpu", device, device)
