
import json
import logging
import os
import sys
import time

from transformers.trainer_callback import ExportableState, TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


def get_logger(name: str = None) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if 'RANK' in os.environ and int(os.environ['RANK']) == 0:
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger


logger = get_logger(__name__)

LOG_FILE_NAME = "trainer_log.jsonl"


class LogCallback(TrainerCallback, ExportableState):
    def __init__(self, start_time: float = None, elapsed_time: float = None):

        self.start_time = time.time() if start_time is None else start_time
        self.elapsed_time = 0 if elapsed_time is None else elapsed_time
        self.last_time = self.start_time

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            if not args.resume_from_checkpoint:
                self.start_time = time.time()
                self.elapsed_time = 0
            else:
                self.start_time = state.stateful_callbacks['LogCallback']['start_time']
                self.elapsed_time = state.stateful_callbacks['LogCallback']['elapsed_time']

        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        self.last_time = time.time()
        if os.path.exists(os.path.join(args.output_dir, LOG_FILE_NAME)) and args.overwrite_output_dir:
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, LOG_FILE_NAME))

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs,
        **kwargs,
    ):
        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        self.elapsed_time += time.time() - self.last_time
        self.last_time = time.time()
        if 'num_input_tokens_seen' in logs:
            logs['num_tokens'] = logs.pop('num_input_tokens_seen')
            state.log_history[-1].pop('num_input_tokens_seen')
            throughput = logs['num_tokens'] / args.world_size / self.elapsed_time
            state.log_history[-1]['throughput'] = logs['throughput'] = throughput
        state.stateful_callbacks["LogCallback"] = self.state()

        logs = dict(
            current_steps=state.global_step,
            total_steps=state.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(state.global_step / state.max_steps * 100, 2) if state.max_steps != 0 else 100,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def state(self) -> dict:
        return {
            'start_time': self.start_time,
            'elapsed_time': self.elapsed_time,
        }

    @classmethod
    def from_state(cls, state):
        return cls(state['start_time'], state['elapsed_time'])
