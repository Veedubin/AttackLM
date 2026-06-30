"""Tests for the AttackLM GUI."""

import pytest
from attacklm_gui.runner import parse_training_line, TrainingMetrics
from attacklm_gui.presets import Preset, BUILTIN_PRESETS


class TestTrainingLineParser:
    """Test parsing of training output lines."""

    def test_parse_loss(self):
        metrics = TrainingMetrics()
        line = "Epoch 1/20   0% | loss 2.6871 | 1,882 tok/s | 0.2 pairs/s | VRAM alloc 5.9 cache 4.8 /15.6 GB"
        result = parse_training_line(line, metrics)
        assert result.loss == 2.6871
        assert result.tok_per_sec == 1882.0
        assert result.pairs_per_sec == 0.2
        assert result.vram_alloc_gb == 5.9
        assert result.vram_cache_gb == 4.8
        assert result.vram_total_gb == 15.6
        assert result.epoch == 1.0
        assert result.total_epochs == 20
        assert result.progress_pct == 0.0

    def test_parse_eval_loss(self):
        metrics = TrainingMetrics()
        line = "Step   158 | Epoch 1.0 | Train Loss: N/A | Eval Loss: 2.4903"
        result = parse_training_line(line, metrics)
        assert result.eval_loss == 2.4903

    def test_parse_trend(self):
        metrics = TrainingMetrics()
        line = "Epoch 2/20   6% | loss 2.5013 | 2,910 tok/s | trend ↓ -0.1516 (0/3)"
        result = parse_training_line(line, metrics)
        assert result.trend == "↓"
        assert result.trend_value == -0.1516

    def test_parse_vram_free(self):
        metrics = TrainingMetrics()
        line = "[GCEpochCallback] Post-eval VRAM: 3.23GB free / 15.57GB total (alloc 5.9GB, cached 5.0GB)"
        result = parse_training_line(line, metrics)
        assert result.vram_free_gb == 3.23
        assert result.vram_total_gb == 15.57

    def test_parse_epoch_progress(self):
        metrics = TrainingMetrics()
        line = "Epoch 12/20  58% | loss 1.1143 | 2,844 tok/s | 0.2 pairs/s"
        result = parse_training_line(line, metrics)
        assert result.epoch == 12.0
        assert result.total_epochs == 20
        assert result.progress_pct == 58.0

    def test_ignore_zero_loss(self):
        """Zero loss (eval step marker) should not overwrite real loss."""
        metrics = TrainingMetrics()
        metrics.loss = 1.5
        line = "Epoch 6/20  25% | loss 0.0000 | 514 tok/s | 0.0 pairs/s"
        result = parse_training_line(line, metrics)
        assert result.loss == 1.5  # unchanged

    def test_loss_history_accumulates(self):
        metrics = TrainingMetrics()
        parse_training_line("loss 2.5 | 1000 tok/s", metrics)
        parse_training_line("loss 2.3 | 1000 tok/s", metrics)
        parse_training_line("loss 2.1 | 1000 tok/s", metrics)
        assert metrics.loss_history == [2.5, 2.3, 2.1]

    def test_eval_loss_history_accumulates(self):
        metrics = TrainingMetrics()
        parse_training_line("Eval Loss: 2.5", metrics)
        parse_training_line("Eval Loss: 2.3", metrics)
        assert metrics.eval_loss_history == [2.5, 2.3]


class TestPresets:
    """Test preset save/load functionality."""

    def test_builtin_presets_exist(self):
        assert len(BUILTIN_PRESETS) >= 4
        names = [p.name for p in BUILTIN_PRESETS]
        assert "3B Q-GaLore Spectrum" in names
        assert "3B LoRA Default" in names

    def test_preset_save_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("attacklm_gui.presets.PRESETS_DIR", tmp_path)

        preset = Preset(
            name="test-preset",
            params={"epochs": 10, "batch_size": 2},
            description="Test preset",
        )
        preset.save()

        loaded = Preset.load("test-preset")
        assert loaded is not None
        assert loaded.name == "test-preset"
        assert loaded.params["epochs"] == 10
        assert loaded.params["batch_size"] == 2

    def test_preset_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr("attacklm_gui.presets.PRESETS_DIR", tmp_path)

        Preset(name="preset-a", params={}).save()
        Preset(name="preset-b", params={}).save()

        names = Preset.list_all()
        assert "preset-a" in names
        assert "preset-b" in names

    def test_preset_delete(self, tmp_path, monkeypatch):
        monkeypatch.setattr("attacklm_gui.presets.PRESETS_DIR", tmp_path)

        Preset(name="to-delete", params={}).save()
        assert Preset.delete("to-delete") is True
        assert Preset.load("to-delete") is None
        assert Preset.delete("nonexistent") is False

    def test_preset_load_nonexistent(self):
        result = Preset.load("definitely-does-not-exist-xyz")
        assert result is None


class TestCommandBuilding:
    """Test that the train form builds correct CLI commands."""

    def test_basic_command(self):
        from attacklm_gui.screens.train_form import TrainFormScreen

        # We can't easily test the full screen, but we can test _build_command
        # by creating a minimal mock
        screen = TrainFormScreen.__new__(TrainFormScreen)

        values = {
            "dataset": "data/test.jsonl",
            "output": "models/test",
            "base_model": "Qwen/Qwen2.5-Coder-3B-Instruct",
            "epochs": "10",
            "batch_size": "1",
            "max_length": "2048",
            "train": True,
        }
        cmd = screen._build_command(values)

        assert "--dataset" in cmd
        assert "data/test.jsonl" in cmd
        assert "--output" in cmd
        assert "models/test" in cmd
        assert "--base-model" in cmd
        assert "--epochs" in cmd
        assert "10" in cmd
        assert "--train" in cmd

    def test_galore_command(self):
        from attacklm_gui.screens.train_form import TrainFormScreen

        screen = TrainFormScreen.__new__(TrainFormScreen)

        values = {
            "dataset": "data/test.jsonl",
            "output": "models/test",
            "use_qgalore": True,
            "galore_rank": "128",
            "spectrum": "0.5",
            "train": True,
        }
        cmd = screen._build_command(values)

        assert "--use-qgalore" in cmd
        assert "--galore-rank" in cmd
        assert "128" in cmd
        assert "--spectrum" in cmd
        assert "0.5" in cmd

    def test_lora_command(self):
        from attacklm_gui.screens.train_form import TrainFormScreen

        screen = TrainFormScreen.__new__(TrainFormScreen)

        values = {
            "dataset": "data/test.jsonl",
            "output": "models/test",
            "lora_r": "32",
            "lora_alpha": "64",
            "use_dora": True,
            "train": True,
        }
        cmd = screen._build_command(values)

        assert "--lora-r" in cmd
        assert "32" in cmd
        assert "--lora-alpha" in cmd
        assert "64" in cmd
        assert "--use-dora" in cmd
        assert "--use-qgalore" not in cmd

    def test_dry_run_no_train_flag(self):
        from attacklm_gui.screens.train_form import TrainFormScreen

        screen = TrainFormScreen.__new__(TrainFormScreen)

        values = {
            "dataset": "data/test.jsonl",
            "output": "models/test",
            "train": False,
        }
        cmd = screen._build_command(values)

        assert "--train" not in cmd
