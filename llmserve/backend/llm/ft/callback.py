from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
import threading
import queue

QUEUE = queue.Queue()
def send_metrics():
    while True:
        item  = QUEUE.get()
        print("============")
        print(item)
        QUEUE.task_done()

threading.Thread(target=send_metrics, daemon=True).start()

class CustomCallback(TrainerCallback):
    """
    Overriding the trainer callback to be able to compute training accuracy as well
    Example taken from:
    https://stackoverflow.com/questions/67457480/how-to-get-the-accuracy-per-epoch-or-step-for-the-huggingface-transformers-train
    """
    METRICS_FILE = "./metrics"

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if control.should_log:
            if len(state.log_history) != 0:
                QUEUE.put(state.log_history[-1])
        return control