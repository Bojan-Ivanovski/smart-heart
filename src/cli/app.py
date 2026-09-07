import typer

from .evaluate import evaluate
from .predict import predict
from .preview import preview
from .train import train


app = typer.Typer(
    name="smartheart",
    help="Train and evaluate the SmartHeart EEG curriculum.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

app.command()(preview)
app.command()(predict)
app.command()(train)
app.command()(evaluate)
