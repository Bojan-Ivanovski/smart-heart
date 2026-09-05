from dataclasses import dataclass
from enum import Enum
from types import ModuleType

import torch


class RuntimeKind(str, Enum):
    AUTO = "auto"
    XLA = "xla"
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


@dataclass(frozen=True)
class Runtime:
    kind: RuntimeKind
    device: object
    model_init_device: object
    xla_model: ModuleType | None = None

    def optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.kind is RuntimeKind.XLA:
            if self.xla_model is None:
                raise RuntimeError("XLA runtime has no torch_xla integration.")
            self.xla_model.optimizer_step(optimizer, barrier=True)
            self.xla_model.mark_step()
            return
        optimizer.step()


def resolve_runtime(requested: RuntimeKind | str = RuntimeKind.AUTO) -> Runtime:
    try:
        kind = RuntimeKind(requested)
    except ValueError as error:
        supported = ", ".join(value.value for value in RuntimeKind)
        raise ValueError(
            f"Unknown runtime '{requested}'. Expected one of: {supported}."
        ) from error

    if kind in {RuntimeKind.AUTO, RuntimeKind.CUDA} and torch.cuda.is_available():
        device = torch.device(RuntimeKind.CUDA.value)
        return Runtime(RuntimeKind.CUDA, device, device)
    if kind is RuntimeKind.CUDA:
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    if kind in {RuntimeKind.AUTO, RuntimeKind.XLA}:
        runtime = _resolve_xla()
        if runtime is not None:
            return runtime
    if kind is RuntimeKind.XLA:
        raise RuntimeError(
            "XLA was requested, but no usable torch_xla device is available."
        )

    mps_available = (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if kind in {RuntimeKind.AUTO, RuntimeKind.MPS} and mps_available:
        device = torch.device(RuntimeKind.MPS.value)
        return Runtime(RuntimeKind.MPS, device, device)
    if kind is RuntimeKind.MPS:
        raise RuntimeError("MPS was requested, but no MPS device is available.")

    device = torch.device(RuntimeKind.CPU.value)
    return Runtime(RuntimeKind.CPU, device, device)


def _resolve_xla() -> Runtime | None:
    try:
        import torch_xla.core.xla_model as xm
    except ImportError:
        return None

    try:
        device = xm.xla_device()
    except RuntimeError:
        return None
    return Runtime(RuntimeKind.XLA, device, RuntimeKind.CPU.value, xm)
