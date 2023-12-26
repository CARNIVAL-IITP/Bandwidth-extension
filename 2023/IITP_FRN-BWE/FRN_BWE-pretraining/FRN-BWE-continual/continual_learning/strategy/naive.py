import pytorch_lightning as pl

class Strategy(pl.Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()

class Naive(Strategy):
    def __init__(self, ):
        super().__init__()
