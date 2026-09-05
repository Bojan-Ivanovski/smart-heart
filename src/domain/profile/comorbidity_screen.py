from ..base import ContractModel
from .sleep_screen import ScreeningStatus, SleepScreen


class ComorbidityScreen(ContractModel):
    anxiety: ScreeningStatus
    depression: ScreeningStatus
    learning_disorder: ScreeningStatus
    sleep: SleepScreen
