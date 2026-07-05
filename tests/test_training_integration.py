#!/usr/bin/env python3
"""Integration smoke test for the training pipeline.

Exercises the critical path: model loading → LoRA application → training → saving.

Run with:
    # Non-GPU tests only (argparse validation)
    python -m pytest tests/test_training_integration.py -v -k "not tiny_model"

    # Full integration (requires GPU + model download)
    RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_training_integration.py -v
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Make the scripts/ dir importable (same pattern as test_balance_buckets.py)
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


class TestTrainingIntegration(unittest.TestCase):
    """Smoke test: load tiny model, train 1 step, verify output."""

    @unittest.skipIf(
        not os.environ.get("RUN_INTEGRATION_TESTS"),
        "Set RUN_INTEGRATION_TESTS=1 to run integration tests (needs GPU + model download)",
    )
    def test_tiny_model_one_step(self):
        """Load tiny-gpt2, train 1 step with QLoRA, verify adapter saved."""
        import torch

        # Skip if no GPU available
        if not torch.cuda.is_available():
            self.skipTest("No CUDA GPU available — skipping GPU-dependent test")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # --- Create a tiny dataset (3 examples, messages format) ---
            dataset_path = tmp / "tiny_dataset.jsonl"
            examples = [
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is persistence in MITRE ATT&CK?",
                        },
                        {
                            "role": "assistant",
                            "content": "Persistence is technique TA0003.",
                        },
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Explain defense evasion."},
                        {"role": "assistant", "content": "Defense evasion is TA0005."},
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "What is privilege escalation?"},
                        {
                            "role": "assistant",
                            "content": "Privilege escalation is TA0004.",
                        },
                    ]
                },
            ]
            with open(dataset_path, "w") as f:
                for ex in examples:
                    f.write(json.dumps(ex) + "\n")

            # --- Import training modules ---
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import get_peft_model, LoraConfig, TaskType
            from trl import SFTTrainer, SFTConfig
            from datasets import load_dataset

            # --- Load tiny model ---
            model_name = "sshleifer/tiny-gpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,  # load directly to GPU
                trust_remote_code=True,
            )
            model = model.cuda()

            # --- Apply LoRA ---
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["c_attn", "c_proj"],  # GPT-2 module names
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # --- Load dataset ---
            dataset = load_dataset("json", data_files=str(dataset_path), split="train")

            # --- Formatting function ---
            def formatting_func(example):
                return tokenizer.apply_chat_template(
                    example["messages"], tokenize=False
                )

            # --- Train 1 step ---
            output_dir = tmp / "output"
            training_args = SFTConfig(
                output_dir=str(output_dir),
                max_steps=1,
                per_device_train_batch_size=1,
                max_length=256,
                logging_steps=1,
                save_strategy="steps",
                save_steps=1,
                save_total_limit=1,
                gradient_checkpointing=False,
                disable_tqdm=True,
                report_to="none",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                dataloader_num_workers=0,
                fp16=False,
                bf16=False,
            )

            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                tokenizer=tokenizer,
                formatting_func=formatting_func,
            )

            trainer.train()

            # --- Verify adapter files exist ---
            adapter_config = output_dir / "adapter_config.json"
            adapter_weights = output_dir / "adapter_model.safetensors"
            self.assertTrue(
                adapter_config.exists(),
                f"Expected {adapter_config} to exist after training",
            )
            self.assertTrue(
                adapter_weights.exists(),
                f"Expected {adapter_weights} to exist after training",
            )

            # Verify adapter_config is valid JSON
            with open(adapter_config) as f:
                cfg = json.load(f)
            self.assertIn("peft_type", cfg)
            self.assertEqual(cfg["peft_type"], "LORA")
            self.assertIn("r", cfg)
            self.assertEqual(cfg["r"], 8)

    def test_argparse_pipeline(self):
        """Verify all training flags parse correctly without running."""
        from train_template import parse_args

        # Minimal required args
        args = parse_args(
            [
                "--dataset",
                "/tmp/nonexistent.jsonl",
                "--output",
                "/tmp/nonexistent_output",
                "--dry-run",
            ]
        )
        self.assertEqual(args.dataset, "/tmp/nonexistent.jsonl")
        self.assertEqual(args.output, "/tmp/nonexistent_output")
        self.assertTrue(args.dry_run)

    def test_argparse_all_flags(self):
        """Verify every training flag parses without error."""
        from train_template import parse_args

        args = parse_args(
            [
                "--dataset",
                "/tmp/d.jsonl",
                "--output",
                "/tmp/out",
                "--base-model",
                "Qwen/Qwen2.5-7B-Instruct",
                "--epochs",
                "10",
                "--batch-size",
                "4",
                "--max-length",
                "1024",
                "--lora-r",
                "32",
                "--lora-alpha",
                "64",
                "--lora-dropout",
                "0.1",
                "--optim",
                "adamw_torch",
                "--save-steps",
                "100",
                "--gradient-accumulation-steps",
                "2",
                "--eval-split",
                "0.2",
                "--early-stopping-patience",
                "5",
                "--train",
                "--packing",
                "--use-rslora",
                "--use-dora",
                "--target-modules",
                "q_proj,v_proj",
                "--max-steps",
                "50",
                "--no-timestamp",
                "--force",
            ]
        )
        self.assertEqual(args.dataset, "/tmp/d.jsonl")
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.lora_r, 32)
        self.assertTrue(args.train)
        self.assertTrue(args.packing)
        self.assertTrue(args.use_dora)
        self.assertEqual(args.target_modules, "q_proj,v_proj")
        self.assertEqual(args.max_steps, 50)
        self.assertTrue(args.no_timestamp)
        self.assertTrue(args.force)

    def test_argparse_dry_run_default(self):
        """Without --train, dry_run should be True (safety default)."""
        from train_template import parse_args

        args = parse_args(
            [
                "--dataset",
                "/tmp/d.jsonl",
                "--output",
                "/tmp/out",
            ]
        )
        # --train not passed → dry_run should be True
        self.assertFalse(args.train)
        # The main() function checks: is_dry_run = args.dry_run or not args.train
        # So without --train, it's a dry run
        self.assertTrue(args.dry_run or not args.train)

    def test_argparse_mutual_exclusion(self):
        """Verify --fp16 and --bf16 are mutually exclusive (argparse handles this)."""
        from train_template import parse_args

        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--dataset",
                    "/tmp/d.jsonl",
                    "--output",
                    "/tmp/out",
                    "--fp16",
                    "--bf16",
                ]
            )

    def test_argparse_resolve_output_path(self):
        """Verify resolve_output_path adds timestamp by default."""
        from train_template import resolve_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            out = resolve_output_path(
                os.path.join(tmpdir, "test_model"),
                no_timestamp=False,
                force=False,
            )
            # Should end in _YYYY-MM-DD_HH-MM
            import re

            self.assertRegex(Path(out).name, r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}$")

    def test_argparse_resolve_output_path_no_timestamp(self):
        """Verify --no-timestamp uses path as-is."""
        from train_template import resolve_output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            out = resolve_output_path(
                os.path.join(tmpdir, "test_model"),
                no_timestamp=True,
                force=False,
            )
            self.assertEqual(Path(out).name, "test_model")


if __name__ == "__main__":
    unittest.main(verbosity=2)
