# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import gc
import os
import ctypes
import ctypes.util
from itertools import chain
from typing import Optional

import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, TrainerCallback

from peft import (
    LoraConfig,
    get_peft_model,
)


class MemoryLoggingCallback(TrainerCallback):
    """Log peak RSS every `log_every_n_steps` training steps."""

    def __init__(self, local_rank, log_every_n_steps=100):
        self.local_rank = local_rank
        self.log_every_n_steps = log_every_n_steps

    def _log_peak_rss(self, step_label):
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        peak_kb = int(line.split()[1])
                        peak_mb = peak_kb / 1024
                        peak_gb = peak_mb / 1024
                        print(f"[MEMORY] Rank {self.local_rank} {step_label} peak RSS: {peak_mb:.0f} MB ({peak_gb:.1f} GB)", flush=True)
                        break
        except Exception:
            pass

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every_n_steps == 0:
            self._log_peak_rss(f"step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        self._log_peak_rss("final")


class StepTimingCallback(TrainerCallback):
    """Diagnose exactly where slow steps spend time + GC monitoring."""

    def __init__(self, local_rank, trainer_ref=None, log_every_n_steps=10):
        self.local_rank = local_rank
        self.log_every_n_steps = log_every_n_steps
        self._step_start = None
        self._step_times = []
        self._gc_count_at_step_start = 0
        self._trainer_ref = trainer_ref

    def on_step_begin(self, args, state, control, **kwargs):
        import time
        self._step_start = time.monotonic()
        self._gc_count_at_step_start = sum(gc.get_stats()[i]["collections"] for i in range(3))

    def on_step_end(self, args, state, control, **kwargs):
        import time
        if self._step_start is None:
            return
        elapsed = time.monotonic() - self._step_start
        self._step_times.append(elapsed)

        gc_count_now = sum(gc.get_stats()[i]["collections"] for i in range(3))
        gc_happened = gc_count_now - self._gc_count_at_step_start

        if self.local_rank == 0:
            gc_str = f" GC={gc_happened}" if gc_happened > 0 else ""
            fwd_bwd = ""
            if self._trainer_ref and hasattr(self._trainer_ref, '_diag_fwd'):
                fwd = self._trainer_ref._diag_fwd
                bwd = self._trainer_ref._diag_bwd
                op = self._trainer_ref._diag_optim
                other = elapsed - fwd - bwd - op
                fwd_bwd = f" fwd={fwd:.2f}s bwd={bwd:.2f}s optim={op:.2f}s other={other:.2f}s"
            print(f"[DIAG] step {state.global_step}: total={elapsed:.2f}s{fwd_bwd}{gc_str}",
                  flush=True)

        if state.global_step % self.log_every_n_steps == 0 and self._step_times:
            recent = self._step_times[-self.log_every_n_steps:]
            avg = sum(recent) / len(recent)
            mn = min(recent)
            mx = max(recent)
            print(f"[TIMING] Rank {self.local_rank} step {state.global_step}: "
                  f"last {len(recent)} steps avg={avg:.2f}s min={mn:.2f}s max={mx:.2f}s",
                  flush=True)


class DiagTrainer(transformers.Trainer):
    """Subclass Trainer to measure forward+backward vs optimizer vs data loading time."""

    def __init__(self, *args, diag_rank=0, fixed_batch_test=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._diag_rank = diag_rank
        self._diag_fwd_bwd = 0
        self._diag_fwd = 0
        self._diag_bwd = 0
        self._diag_optim = 0
        self._fixed_batch_test = fixed_batch_test
        self._cached_batch = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        import time

        # Fixed batch test: cache first batch, reuse for all steps
        if self._fixed_batch_test:
            if self._cached_batch is None:
                self._cached_batch = {k: v.clone() for k, v in inputs.items()}
            else:
                inputs = {k: v.clone() for k, v in self._cached_batch.items()}

        model.train()
        inputs = self._prepare_inputs(inputs)

        # Forward
        t0 = time.monotonic()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        del inputs
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if num_items_in_batch is None and self.compute_loss_func is None:
            loss = loss / self.current_gradient_accumulation_steps
        self._diag_fwd = time.monotonic() - t0

        # Backward (includes DDP allreduce)
        t1 = time.monotonic()
        self.accelerator.backward(loss)
        self._diag_bwd = time.monotonic() - t1

        self._diag_fwd_bwd = self._diag_fwd + self._diag_bwd
        return loss.detach()

    def _inner_training_loop(self, *args, **kwargs):
        # Wrap optimizer.step to time it - but optimizer is created inside
        # super()._inner_training_loop, so we patch create_optimizer instead
        original_create_optimizer = self.create_optimizer

        def patched_create_optimizer():
            original_create_optimizer()
            import time
            original_step = self.optimizer.step

            def timed_step(*a, **kw):
                t0 = time.monotonic()
                result = original_step(*a, **kw)
                self._diag_optim = time.monotonic() - t0
                return result

            self.optimizer.step = timed_step

        self.create_optimizer = patched_create_optimizer
        return super()._inner_training_loop(*args, **kwargs)


def train(
    base_model: str = "path/to/model",
    data_path: str = "yahma/alpaca-cleaned",
    output_dir: str = "olora",
    batch_size: int = 16,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 1024,
    val_set_size: int = 0,
    quantize: bool = False,
    eval_step: int = 100,
    save_step: int = 100,
    device_map: str = "auto",
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: list[str] = None,
    dtype: str = "float16",
    init_lora_weights="olora",
    seed: Optional[int] = None,
    group_texts_enabled: bool = True,
    max_steps: int = -1,
    fixed_batch_test: bool = False,
    use_torch_compile: bool = False,
):
    # Per-rank NUMA binding: 43 physical cores per rank (no HT)
    local_rank = int(
        os.environ.get("MPI_LOCALRANKID", 0) or
        os.environ.get("LOCAL_RANK", 0) or
        os.environ.get("PMI_RANK", 0) or 0
    )
    NUMA_CPUS = {
        0: list(range(0, 43)),
        1: list(range(43, 86)),
        2: list(range(86, 129)),
        3: list(range(129, 172)),
        4: list(range(172, 215)),
        5: list(range(215, 258)),
        6: list(range(258, 301)),
        7: list(range(301, 344)),
    }
    if local_rank in NUMA_CPUS:
        os.sched_setaffinity(0, NUMA_CPUS[local_rank])
        try:
            libnuma_path = ctypes.util.find_library("numa")
            if libnuma_path:
                libnuma = ctypes.CDLL(libnuma_path)
                libnuma.numa_set_preferred(local_rank)
                libnuma.numa_parse_nodestring.restype = ctypes.c_void_p
                libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
                bitmask = libnuma.numa_parse_nodestring(str(local_rank).encode())
                if bitmask:
                    libnuma.numa_set_membind(bitmask)
                    libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]
                    libnuma.numa_bitmask_free(bitmask)
            print(f"[Rank {local_rank}] Bound to NUMA node {local_rank} "
                  f"(43 cores, membind)", flush=True)
        except Exception as e:
            print(f"[Rank {local_rank}] NUMA membind failed: {e}", flush=True)

    # Set device_map to the right place when enabling DDP.
    world_size = int(os.environ.get("WORLD_SIZE", 0)) or int(os.environ.get("PMI_SIZE", 0))
    if world_size > 1 and device_map != "cpu":
        from accelerate import Accelerator

        device_map = {"": Accelerator().process_index}
    # Set seed
    if seed is not None:
        set_seed(seed)
    model_kwargs = {"dtype": getattr(torch, dtype), "device_map": device_map}
    if quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # For some tokenizer with no pad token like llama
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=not group_texts_enabled,
            max_length=cutoff_len if not group_texts_enabled else None,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(example):
        full_prompt = generate_prompt(example)
        tokenized_full_prompt = tokenize(full_prompt)
        return tokenized_full_prompt

    # Concatenate all texts and split into chunks of cutoff_len.
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[-1]])
        result = {
            k: [t[i : i + cutoff_len] for i in range(0, total_length, cutoff_len)]
            for k, t in concatenated_examples.items()
        }
        return result

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=init_lora_weights,
    )
    model = get_peft_model(model, config)

    if use_torch_compile:
        print(f"[Rank {local_rank}] Applying torch.compile...", flush=True)
        model = torch.compile(model)

    if data_path.endswith(".json") or data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_path)
    else:
        data = load_dataset(data_path)

    if val_set_size > 0:
        train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
        train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    else:
        train_data = data["train"].map(generate_and_tokenize_prompt)
        if group_texts_enabled:
            train_data = train_data.map(group_texts, batched=True)
        val_data = None

    # Keep only model-relevant columns
    keep_cols = {"input_ids", "attention_mask", "labels"}
    remove_cols = [c for c in train_data.column_names if c not in keep_cols]
    if remove_cols:
        train_data = train_data.remove_columns(remove_cols)

    memory_cb = MemoryLoggingCallback(local_rank, log_every_n_steps=100)
    timing_cb = StepTimingCallback(local_rank, log_every_n_steps=10)

    trainer = DiagTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        callbacks=[memory_cb, timing_cb],
        diag_rank=local_rank,
        fixed_batch_test=fixed_batch_test,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=batch_size,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=100,
            optim="adamw_torch",
            eval_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=eval_step if val_set_size > 0 else None,
            save_steps=save_step,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if world_size > 1 else None,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    timing_cb._trainer_ref = trainer
    gc.disable()
    trainer.train()
    gc.enable()

    model.save_pretrained(output_dir)


def generate_prompt(example):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
            ### Instruction:
            {example["instruction"]}
            ### Response:
            {example["output"]}"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="path/to/model")
    parser.add_argument("--data_path", type=str, default="yahma/alpaca-cleaned")
    parser.add_argument("--output_dir", type=str, default="olora")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--cutoff_len", type=int, default=1024)
    parser.add_argument("--val_set_size", type=int, default=0)
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--eval_step", type=int, default=100)
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--init_lora_weights", type=str, default="olora")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--group_texts", action="store_true", default=True)
    parser.add_argument("--no_group_texts", action="store_false", dest="group_texts")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--fixed_batch_test", action="store_true", help="Reuse first batch for all steps (perf diagnosis)")
    parser.add_argument("--torch_compile", action="store_true", help="Use torch.compile for operator fusion")

    args = parser.parse_args()

    train(
        base_model=args.base_model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        cutoff_len=args.cutoff_len,
        val_set_size=args.val_set_size,
        quantize=args.quantize,
        eval_step=args.eval_step,
        save_step=args.save_step,
        device_map=args.device_map,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        dtype=args.dtype,
        init_lora_weights=args.init_lora_weights,
        seed=args.seed,
        group_texts_enabled=args.group_texts,
        max_steps=args.max_steps,
        fixed_batch_test=args.fixed_batch_test,
        use_torch_compile=args.torch_compile,
    )
