import pytorch_lightning as pl


class Strategy(pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()
