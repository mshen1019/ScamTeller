"""LoRA fine-tune of Qwen3-8B on per-decision-point verdicts (records from scamteller.data).

Each record is [user = rubric prompt + transcript so far, assistant = "A" | "B"]. Loss is applied
only to the assistant reply. The script asserts that the rendered generation prompt is an exact
token prefix of the rendered sample, using the base model's own chat template
(enable_thinking=False), so training sees exactly the position that serving reads the score from.

Hyperparameters come from a JSON config (configs/qwen3_8b_prefix_ift.json); command-line flags
override it.

Launch (8 GPUs, DDP):
  torchrun --nproc_per_node 8 -m scamteller.train --config configs/qwen3_8b_prefix_ift.json \
      --train data/train.jsonl --val data/eval.jsonl --out runs/scamteller

Outputs in --out:
  adapter/        final LoRA adapter (bf16)
  merged/         base + adapter merged, if --merge
  MODEL_CARD.md   base, data, hyperparameters, final validation loss
"""
from __future__ import annotations

import argparse
import inspect
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
HPARAMS = ["base", "base_revision", "max_len", "epochs", "lr", "per_device_bs", "grad_accum",
           "warmup_ratio", "lora_r", "lora_alpha", "lora_dropout", "seed"]


def render(tok, msgs, **kw):
    return tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False, **kw)


def build_samples(path, tok, max_len):
    samples, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            msgs = json.loads(line)["messages"]
            prompt = render(tok, msgs[:-1], add_generation_prompt=True)
            full = render(tok, msgs)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            f_ids = tok(full, add_special_tokens=False)["input_ids"]
            if not full.startswith(prompt) or f_ids[:len(p_ids)] != p_ids:
                raise ValueError("chat template is not prefix-consistent; training would not "
                                 "match the position the score is read from at serving time")
            if len(f_ids) > max_len:
                skipped += 1
                continue
            samples.append({"input_ids": f_ids, "labels": [-100] * len(p_ids) + f_ids[len(p_ids):]})
    return samples, skipped


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda xs, v: xs + [v] * (n - len(xs))
        return {"input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch]),
                "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch])}


def make_training_args(**kw):
    """TrainingArguments across transformers 4.x and 5.x: rename or drop keys a version lacks."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "evaluation_strategy" in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params]
    if dropped:
        print(f"[args] this transformers version lacks {dropped}; continuing without them", flush=True)
    return TrainingArguments(**{k: v for k, v in kw.items() if k in params})


def load_config(args):
    cfg = json.load(open(args.config)) if args.config else {}
    for k in HPARAMS:
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    missing = [k for k in HPARAMS if cfg.get(k) is None and k != "base_revision"]
    if missing:
        raise SystemExit(f"missing hyperparameters (set them in --config or on the command line): {missing}")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--merge", action="store_true", help="also write base + adapter merged")
    for k in HPARAMS:
        ap.add_argument("--" + k.replace("_", "-"), dest=k, default=None,
                        type=str if k in ("base", "base_revision") else float if k in
                        ("epochs", "lr", "warmup_ratio", "lora_dropout") else int)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke tests only")
    ap.add_argument("--no-bf16", action="store_true", help="CPU smoke tests only")
    args = ap.parse_args()
    cfg = load_config(args)
    set_seed(cfg["seed"])
    bf16 = not args.no_bf16

    tok = AutoTokenizer.from_pretrained(cfg["base"], revision=cfg.get("base_revision"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    train, skip_tr = build_samples(args.train, tok, cfg["max_len"])
    val, skip_va = build_samples(args.val, tok, cfg["max_len"])
    print(f"[data] train={len(train)} (skipped >{cfg['max_len']} tokens: {skip_tr}) "
          f"val={len(val)} (skipped: {skip_va})", flush=True)

    dtype = torch.bfloat16 if bf16 else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["base"], revision=cfg.get("base_revision"), dtype=dtype)
    except TypeError:  # transformers < 4.56
        model = AutoModelForCausalLM.from_pretrained(cfg["base"], revision=cfg.get("base_revision"),
                                                     torch_dtype=dtype)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = make_training_args(
        output_dir=args.out, num_train_epochs=cfg["epochs"], max_steps=args.max_steps,
        per_device_train_batch_size=cfg["per_device_bs"], per_device_eval_batch_size=cfg["per_device_bs"],
        gradient_accumulation_steps=cfg["grad_accum"], learning_rate=cfg["lr"],
        lr_scheduler_type="cosine", warmup_ratio=cfg["warmup_ratio"], weight_decay=0.0, bf16=bf16,
        logging_steps=10, eval_strategy="no", save_strategy="no", group_by_length=True,
        report_to="none", ddp_find_unused_parameters=False, remove_unused_columns=False,
        seed=cfg["seed"], dataloader_num_workers=2)
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()
    metrics = trainer.evaluate()

    if not trainer.is_world_process_zero():
        return
    adapter_dir = os.path.join(args.out, "adapter")
    trainer.model.save_pretrained(adapter_dir, safe_serialization=True)
    tok.save_pretrained(adapter_dir)
    if args.merge:
        merged = trainer.model.merge_and_unload()
        merged.to(torch.bfloat16).save_pretrained(os.path.join(args.out, "merged"), safe_serialization=True)
        tok.save_pretrained(os.path.join(args.out, "merged"))
    card = f"""# Model card: {os.path.basename(os.path.abspath(args.out))}

- **Base:** `{cfg['base']}` (revision `{cfg.get('base_revision')}`)
- **Data:** train `{args.train}` ({len(train)} decision points), val `{args.val}` ({len(val)})
- **Format:** rubric prompt + transcript so far -> `A` (scam) / `B`; loss on the reply only;
  `enable_thinking=False`; max_len {cfg['max_len']}
- **Score at serving:** P(A) / (P(A) + P(B)) at the first reply token
- **LoRA:** r={cfg['lora_r']}, alpha={cfg['lora_alpha']}, dropout={cfg['lora_dropout']}, targets={TARGETS}
- **Optim:** lr={cfg['lr']}, cosine, warmup {cfg['warmup_ratio']}, epochs={cfg['epochs']},
  per-device bs={cfg['per_device_bs']}, grad_accum={cfg['grad_accum']}, world size={trainer.args.world_size},
  seed={cfg['seed']}
- **Final validation loss:** {metrics.get('eval_loss')}
"""
    with open(os.path.join(args.out, "MODEL_CARD.md"), "w") as f:
        f.write(card)
    print(f"[done] adapter -> {adapter_dir}", flush=True)


if __name__ == "__main__":
    main()
