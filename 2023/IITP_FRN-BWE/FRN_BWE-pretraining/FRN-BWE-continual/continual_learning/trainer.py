from typing import Any

import pytorch_lightning as pl
# from pytorch_lightning.loops.fit_loop import _FitLoop
# from pytorch_lightning.loops import fit_loop
from pytorch_lightning.loops import FitLoop
# from pytorch_lightning.trainer.trai
# from pytorch_lightning.loops.loop import l

class ContinualTrainer(pl.Trainer):

    def __init__(self, tasks: int = 0, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.task_id: int = 0
        self.tasks: int = tasks
        self.fit_loop = ContinualFitLoop(max_epochs=kwargs['max_epochs'])


class ContinualFitLoop(FitLoop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any):
        self.reset()

        self.on_run_start()

        self.trainer.should_stop = False

        while not self.done:
            try:
                self.on_advance_start()
                self.advance()
                self.on_advance_end()
                self._restarting = False
            except StopIteration:
                break
        self._restarting = False

        self.on_run_end()
