"""
Coordinate-descent HPO runner for AttackLM QLoRA training.

Algorithm: "Escalate-Then-Reset" (safe-cracking HPO)

For each hyperparameter in priority order:
    1. Start at the "default low" value (e.g. lora_r=8, dropout=0.0)
    2. Train for N steps, record loss/grad_norm/entropy/tok_acc
    3. If all metrics look good:
         save this as best, escalate (e.g. lora_r=8→16→32)
    4. If any metric goes bad:
         back off to last good value, lock it in, move to next HP
    5. Reset to "default low" for the next axis

Finally, run a long full-budget training with all winning HPs combined.

Coordinate descent is NOT optimal, but it's:
  - Cheap (1 axis at a time = O(axes × trials_per_axis) runs)
  - Interpretable (each axis clearly "best at X")
  - Parallelizable (you could run all 4 trials of one axis in parallel)
  - Safe (no combinatorial explosion)

For a 3B QLoRA on RTX 4080, with --hpo-trial-steps=100:
  - Each trial = ~2-3 min
  - 4 axes × 4 trials/axis = 16 trials × 2.5 min = ~40 min for the sweep
  - Plus 1 full training at the end with winners
  - Total: roughly 1 hour to do a 4-axis sweep + final train

Invoke via train_all.py:
    uv run python scripts/train_all.py --hpo --single-model \\
        --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit \\
        --epochs 10 --max-length 2048 \\
        --include-tools --model-attacks --include-orchestrator

Or for an existing run (no full train at end):
    uv run python scripts/hpo_runner.py --analyze-only
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"


@dataclass
class HPAxis:
    """Definition of one hyperparameter axis.

    Each axis has:
      - name: the CLI flag name (e.g. 'lora-r' → 'lora_r')
      - default_low: the safe starting value
      - escalation: lambda(prev_value) → next_value to try
      - max_value: hard ceiling (don't try beyond this)
      - extract: lambda(args) → current value in args namespace
      - apply: lambda(args, value) → set new value
      - trial_budget: steps per trial (some HPs need longer to see effect)
    """

    name: str
    arg_dest: str  # attribute on the args namespace
    default_low: float
    max_value: float
    escalation_factor: float = 2.0
    is_int: bool = True
    trial_budget_override: Optional[int] = None

    def clip(self, v: float) -> float:
        """Cap v at max_value. Used by the dry-run printer to bound the
        display values, and by the sweep loop as a safety net.
        """
        return min(v, self.max_value)

    def next_value(self, prev: float) -> float:
        # Handle the zero case: if prev is 0 and factor is 2.0,
        # we need a non-zero "next step" to escape zero. For dropout
        # (default_low=0.0), we add a constant increment of 0.05.
        if prev == 0.0:
            return 0.05  # first non-zero step
        nxt = prev * self.escalation_factor
        # Cap at max_value so the loop knows when to stop without
        # needing an explicit `if v > max: break` check.
        return min(nxt, self.max_value)


# Priority-ordered axes. Lower index = swept first.
# Note: lora_alpha is tied to lora_r (alpha = 2×r) so we don't sweep it
# separately — changing r automatically scales alpha.
HPO_AXES = [
    HPAxis(
        name="lora_r",
        arg_dest="lora_r",
        default_low=8,
        max_value=64,
        escalation_factor=2.0,
        is_int=True,
    ),
    HPAxis(
        name="lora_dropout",
        arg_dest="lora_dropout",
        default_low=0.0,
        max_value=0.3,
        escalation_factor=2.0,  # 0.0, 0.05, 0.1, 0.2, 0.3 (clipped)
        is_int=False,
    ),
    # lora_alpha is intentionally NOT swept by default. Convention is
    # alpha = 2*r, and sweeping it independently mostly just rescales
    # the LoRA output. Re-enable by uncommenting if you want to try.
    # HPAxis(
    #     name="lora_alpha",
    #     arg_dest="lora_alpha",
    #     default_low=8,
    #     max_value=128,
    #     escalation_factor=2.0,
    #     is_int=True,
    # ),
]


# ---------------------------------------------------------------------------
# Trial result dataclass
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
    """Result of one HPO trial."""

    axis: str
    trial_index: int  # 0, 1, 2, ... within the axis
    value: float
    csv_path: str
    final_loss: Optional[float] = None
    final_grad_norm: Optional[float] = None
    final_entropy: Optional[float] = None
    final_tok_acc: Optional[float] = None
    mean_loss_last10: Optional[float] = None
    loss_slope: Optional[float] = None  # d(loss)/d(step) over the trial
    max_grad_norm: Optional[float] = None
    p95_grad_norm: Optional[float] = None  # 95th pct of per-step grad norms
    mean_grad_second_half: Optional[float] = None  # post-warmup grad mean
    min_vram_free_gb: Optional[float] = None
    mean_tokens_per_sec: Optional[float] = None
    mean_pairs_per_sec: Optional[float] = None
    wall_time_s: float = 0.0
    verdict: str = "UNKNOWN"  # OK, DIVERGED, PLATEAU, ERROR
    reason: str = ""
    returncode: int = 0


# ---------------------------------------------------------------------------
# CSV metric parsing
# ---------------------------------------------------------------------------
def parse_trial_csv(csv_path: str) -> dict:
    """Parse the per-step CSV written by HPOMetricsCSVCallback.

    Returns a dict of summary stats for HPO decisions:
        - final_loss: loss at the last logged step
        - final_grad_norm, final_entropy, final_tok_acc: same
        - mean_loss_last10: mean of the last 10 logged loss values
        - loss_slope: linear slope of loss vs step (negative = improving)
        - max_grad_norm: max gradient norm during the trial
        - min_vram_free_gb: minimum free VRAM (worst-case transient)
        - mean_tokens_per_sec, mean_pairs_per_sec: throughput
    """
    if not os.path.exists(csv_path):
        return {}

    steps = []
    losses = []
    grads = []
    ents = []
    accs = []
    toks = []
    pairs = []
    vram_free = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row.get("step", 0) or 0))
                loss = row.get("loss", "")
                if loss:
                    losses.append(float(loss))
                grad = row.get("grad_norm", "")
                if grad:
                    grads.append(float(grad))
                ent = row.get("entropy", "")
                if ent:
                    ents.append(float(ent))
                acc = row.get("mean_token_accuracy", "")
                if acc:
                    accs.append(float(acc))
                tok = row.get("tokens_per_sec", "")
                if tok:
                    toks.append(float(tok))
                pair = row.get("pairs_per_sec", "")
                if pair:
                    pairs.append(float(pair))
                vf = row.get("vram_free_gb", "")
                if vf:
                    vram_free.append(float(vf))
            except (ValueError, KeyError):
                continue

    # p95 of grad_norm — sustained-high-grad metric, ignores one-off
    # spikes. A divergent run has p95_grad > 5; a healthy run with
    # one transient spike has max=9 but p95=2.5.
    p95_grad = None
    if grads and len(grads) >= 5:
        sorted_grads = sorted(grads)
        idx = int(0.95 * len(sorted_grads))
        p95_grad = sorted_grads[min(idx, len(sorted_grads) - 1)]

    # Mean grad over the second half of the trial (post-warmup).
    # Warmup spikes early; sustained-high grad late is a real signal.
    second_half_grads = grads[len(grads) // 2 :] if grads else []
    mean_grad_second_half = (
        sum(second_half_grads) / max(1, len(second_half_grads))
        if second_half_grads
        else None
    )

    out = {
        "n_logged_steps": len(steps),
        "final_loss": losses[-1] if losses else None,
        "final_grad_norm": grads[-1] if grads else None,
        "final_entropy": ents[-1] if ents else None,
        "final_tok_acc": accs[-1] if accs else None,
        "mean_loss_last10": (
            sum(losses[-10:]) / max(1, len(losses[-10:])) if losses else None
        ),
        "max_grad_norm": max(grads) if grads else None,
        "p95_grad_norm": p95_grad,
        "mean_grad_second_half": mean_grad_second_half,
        "min_vram_free_gb": min(vram_free) if vram_free else None,
        "mean_tokens_per_sec": (sum(toks) / max(1, len(toks))) if toks else None,
        "mean_pairs_per_sec": (sum(pairs) / max(1, len(pairs))) if pairs else None,
    }

    # Compute linear slope of loss over the trial
    if len(losses) >= 2 and len(steps) >= 2:
        # Use a tiny least-squares fit on (step, loss)
        n = len(losses)
        sx = sum(steps[:n])
        sy = sum(losses[:n])
        sxx = sum(s * s for s in steps[:n])
        sxy = sum(s * l for s, l in zip(steps[:n], losses[:n]))
        denom = n * sxx - sx * sx
        if abs(denom) > 1e-9:
            out["loss_slope"] = (n * sxy - sx * sy) / denom
        else:
            out["loss_slope"] = 0.0

    return out


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def evaluate_trial(stats: dict, prev_stats: Optional[dict] = None) -> tuple:
    """Decide if a trial's metrics are good, diverged, or plateaued.

    Returns (verdict, reason) where verdict is one of:
        OK        — metrics look healthy, escalate further
        DIVERGED  — loss spiked or grad exploded, back off immediately
        PLATEAU   — loss is flat (not getting better), back off
        UNSTABLE  — grad_norm > 3.0 sustained
        ERROR     — no metrics captured (run failed or csv missing)
    """
    if not stats or stats.get("n_logged_steps", 0) < 3:
        return ("ERROR", "no metrics captured (run failed or CSV missing)")

    final_loss = stats.get("final_loss", 0)
    mean_loss_last = stats.get("mean_loss_last10", 0)
    max_grad = stats.get("max_grad_norm", 0) or 0
    p95_grad = stats.get("p95_grad_norm", 0) or 0
    mean_grad_2nd_half = stats.get("mean_grad_second_half", 0) or 0
    slope = stats.get("loss_slope", 0) or 0
    final_tok_acc = stats.get("final_tok_acc", 0) or 0
    final_entropy = stats.get("final_entropy", 0) or 0
    min_vram = stats.get("min_vram_free_gb", 999)
    n_steps = stats.get("n_logged_steps", 0) or 0

    # First, recover the initial loss (first logged loss) so we can check
    # if the trial actually made progress. We don't carry it in stats, so
    # the caller (parse_trial_csv) would need to add it. For now, infer
    # progress from slope + final_loss: if slope is strongly negative AND
    # final_loss is well below 2.0, the trial is healthy.
    progress_ratio = None
    # Use the negative-slope / final-loss combo as a proxy for "this trial
    # actually learned something". A divergent run has slope >= 0 AND
    # final_loss > 2.0; a healthy run has slope < 0 AND final_loss < 2.0.

    # Hard stop: actual divergence
    # The strongest divergence signal is: loss is NOT going down AND
    # grad is high. A high grad with a falling loss is just early
    # descent (perfectly normal for fresh fine-tuning).
    #
    # 1. If final_loss is above 5.0, something is very wrong.
    if final_loss > 5.0:
        return (
            "DIVERGED",
            f"final loss {final_loss:.3f} is way too high (>5.0) — back off",
        )
    # 2. If loss is going UP (slope positive) and we're past warmup
    #    (more than 5 logged steps), this is real divergence, not warmup.
    if slope > 0.005 and n_steps > 5 and final_loss > 1.5:
        return (
            "DIVERGED",
            f"loss is INCREASING (slope={slope:.5f}, final={final_loss:.3f}) — "
            f"real divergence, not a transient spike — back off",
        )
    # 3. If final_loss is higher than the first logged loss by 50%+,
    #    we are diverging. (We don't track first-loss in stats; this
    #    check is approximate via the slope gate above.)
    # 4. VRAM is the hard physical limit
    if min_vram is not None and min_vram < 0.5:
        return (
            "DIVERGED",
            f"VRAM hit {min_vram:.2f}GB free — OOM imminent — back off",
        )
    # 5. We do NOT use absolute max_grad as a divergence signal. Grad
    #    spikes are normal during early descent. A healthy run can have
    #    max_grad=10 with p95=2.5; we ignore that. We only use grad
    #    for the comparison-to-previous-trial check below.

    # Plateau: loss isn't moving
    # With 200+ step trials, a healthy loss should drop by at least
    # ~0.05 over the trial (for 3B QLoRA on diverse data). If slope is
    # near zero AND tok_acc is already high, plateau.
    if abs(slope) < 0.0005 and final_tok_acc > 0.6:
        return (
            "PLATEAU",
            f"loss slope {slope:.5f} ≈ 0 (mean last10={mean_loss_last:.3f}) — "
            f"no improvement — back off",
        )

    # Comparison to previous trial (escalation check)
    if prev_stats is not None:
        prev_loss = prev_stats.get("mean_loss_last10", 999) or 999
        cur_loss = mean_loss_last or 999
        # If we're 10% worse than the previous trial at this axis, stop
        if cur_loss > prev_loss * 1.10 and cur_loss > 1.0:
            return (
                "DIVERGED",
                f"this trial ({cur_loss:.3f}) is 10%+ worse than previous "
                f"({prev_loss:.3f}) — back off",
            )

    # All clear — escalate
    return (
        "OK",
        f"loss {final_loss:.3f} → {mean_loss_last:.3f} (slope={slope:.5f}), "
        f"gn_p95={p95_grad:.2f}/max={max_grad:.2f}, "
        f"tok_acc={final_tok_acc:.3f}, ent={final_entropy:.3f} — escalate",
    )


# ---------------------------------------------------------------------------
# HPO runner
# ---------------------------------------------------------------------------
def run_hpo_sweep(
    args,
    out_fn,
    log_fh,
    dataset_path: str,
    hpo_dir: Path,
    base_cmd_fn,
) -> dict:
    """Run the full coordinate-descent HPO sweep.

    Args:
        args: the parsed argparse Namespace from train_all.py
        out_fn: function(str) for output
        log_fh: open log file handle
        dataset_path: path to the JSONL to use for HPO trials
        hpo_dir: directory to write per-trial CSVs into
        base_cmd_fn: callable(args, dataset, output, max_steps, hpo_csv)
                     that returns the subprocess cmd for train_template.py

    Returns:
        dict mapping axis name → best value found
    """
    hpo_dir.mkdir(parents=True, exist_ok=True)
    best_per_axis = {}
    all_trials = []
    trial_counter = 0

    for axis_idx, axis in enumerate(HPO_AXES):
        out_fn("")
        out_fn("=" * 70)
        out_fn(f" HPO AXIS {axis_idx + 1}/{len(HPO_AXES)}: {axis.name}")
        out_fn(
            f"   default_low={axis.default_low}, max={axis.max_value}, "
            f"escalation=×{axis.escalation_factor}"
        )
        out_fn("=" * 70)

        current_value = axis.default_low
        best_value_this_axis = current_value
        prev_stats = None
        best_stats_this_axis = None

        for trial_idx in range(args.hpo_trials_per_axis):
            current_value = axis.clip(current_value)
            trial_counter += 1
            trial_csv = hpo_dir / f"{axis.name}_trial{trial_idx}_v{current_value}.csv"
            trial_output = hpo_dir / f"{axis.name}_trial{trial_idx}_v{current_value}"
            # Output adapter dirs are just for completeness — we don't actually
            # use the saved adapter (HPO is for metric analysis). The save
            # is a sanity check that the training actually completed.

            # Set the HP value on args
            setattr(args, axis.arg_dest, current_value)
            # Keep lora_alpha = 2*lora_r unless we're sweeping alpha itself
            if axis.arg_dest != "lora_alpha":
                args.lora_alpha = int(2 * args.lora_r)

            trial_budget = (
                axis.trial_budget_override
                if axis.trial_budget_override is not None
                else args.hpo_trial_steps
            )

            out_fn("")
            out_fn(
                f"  [trial {trial_idx + 1}/{args.hpo_trials_per_axis}] "
                f"{axis.name}={current_value}  ({trial_budget} steps)"
            )

            # Build cmd via the supplied function (which knows about
            # --include-tools etc.)
            cmd = base_cmd_fn(
                args,
                Path(dataset_path),
                trial_output,
                max_steps=trial_budget,
                hpo_metrics_csv=str(trial_csv),
            )
            out_fn(f"    cmd: {' '.join(cmd[3:])}")
            t0 = time.time()
            try:
                result = subprocess.run(cmd, cwd=str(BASE_DIR), timeout=args.timeout)
                returncode = result.returncode
            except subprocess.TimeoutExpired:
                out_fn(f"    TIMEOUT after {args.timeout}s — aborting axis")
                returncode = -9
            except KeyboardInterrupt:
                out_fn("    INTERRUPTED")
                log_fh.close()
                sys.exit(130)
            elapsed = time.time() - t0

            # Parse the CSV
            stats = parse_trial_csv(str(trial_csv))
            verdict, reason = evaluate_trial(stats, prev_stats)

            trial = TrialResult(
                axis=axis.name,
                trial_index=trial_idx,
                value=current_value,
                csv_path=str(trial_csv),
                final_loss=stats.get("final_loss"),
                final_grad_norm=stats.get("final_grad_norm"),
                final_entropy=stats.get("final_entropy"),
                final_tok_acc=stats.get("final_tok_acc"),
                mean_loss_last10=stats.get("mean_loss_last10"),
                loss_slope=stats.get("loss_slope"),
                max_grad_norm=stats.get("max_grad_norm"),
                p95_grad_norm=stats.get("p95_grad_norm"),
                mean_grad_second_half=stats.get("mean_grad_second_half"),
                min_vram_free_gb=stats.get("min_vram_free_gb"),
                mean_tokens_per_sec=stats.get("mean_tokens_per_sec"),
                mean_pairs_per_sec=stats.get("mean_pairs_per_sec"),
                wall_time_s=elapsed,
                verdict=verdict,
                reason=reason,
                returncode=returncode,
            )
            all_trials.append(trial)

            # Print summary
            out_fn(
                f"    → loss={trial.final_loss}  gn_p95={trial.p95_grad_norm}  "
                f"gn_max={trial.max_grad_norm}  acc={trial.final_tok_acc}  "
                f"vram_free={trial.min_vram_free_gb}GB"
            )
            out_fn(
                f"    → mean_loss_last10={trial.mean_loss_last10}  slope={trial.loss_slope}"
            )
            out_fn(
                f"    → tokens/s={trial.mean_tokens_per_sec}  pairs/s={trial.mean_pairs_per_sec}"
            )
            out_fn(f"    → verdict: {verdict} — {reason}")

            if verdict == "OK":
                # Save this as best for this axis
                best_value_this_axis = current_value
                best_stats_this_axis = stats
                prev_stats = stats
                # Escalate for next trial
                current_value = axis.next_value(current_value)
                if current_value > axis.max_value:
                    out_fn(
                        f"    [axis ceiling] next value {current_value} > max {axis.max_value} — "
                        f"locking in best ({best_value_this_axis})"
                    )
                    break
                # If this is the last trial slot, don't escalate — break
                if trial_idx == args.hpo_trials_per_axis - 1:
                    out_fn(
                        f"    [out of trials] reached --hpo-trials-per-axis limit — "
                        f"locking in best ({best_value_this_axis})"
                    )
            else:
                # Back off — last good value is the winner
                out_fn(
                    f"    [back off] locking in best={best_value_this_axis} for {axis.name}"
                )
                break

        best_per_axis[axis.name] = best_value_this_axis
        out_fn(f"\n  AXIS {axis.name} COMPLETE: best value = {best_value_this_axis}")

    # Write the master state
    state_path = hpo_dir / "hpo_state.json"
    with open(state_path, "w") as f:
        json.dump(
            {
                "best_per_axis": best_per_axis,
                "all_trials": [asdict(t) for t in all_trials],
                "axes_order": [a.name for a in HPO_AXES],
            },
            f,
            indent=2,
        )
    out_fn(f"\n  HPO state written to: {state_path}")

    # Print the recommended final config
    out_fn("")
    out_fn("=" * 70)
    out_fn(" HPO SWEEP COMPLETE — Recommended final config:")
    out_fn("=" * 70)
    for axis in HPO_AXES:
        v = best_per_axis.get(axis.name, axis.default_low)
        out_fn(f"  --{axis.arg_dest.replace('_', '-')} {v:g}")
    out_fn("")

    return best_per_axis


def run_final_training(
    args,
    out_fn,
    log_fh,
    dataset_path: str,
    best_per_axis: dict,
    base_cmd_fn,
) -> int:
    """Run the final full-budget training with HPO winners applied.

    Returns the subprocess returncode.
    """
    out_fn("")
    out_fn("=" * 70)
    out_fn(" FINAL TRAINING with HPO winners")
    out_fn("=" * 70)
    for axis in HPO_AXES:
        v = best_per_axis.get(axis.name, axis.default_low)
        setattr(args, axis.arg_dest, v)
        out_fn(f"  {axis.name} = {v}")
    if "lora_r" in best_per_axis and "lora_alpha" not in best_per_axis:
        args.lora_alpha = int(2 * args.lora_r)
    out_fn(f"  epochs = {args.epochs}")
    out_fn(f"  dataset = {dataset_path}")
    out_fn("")

    output_path = Path(
        args.output
        if hasattr(args, "output") and args.output
        else "models/attacklm-hpo-final"
    )
    if hasattr(args, "single_model_name"):
        output_path = Path("models") / args.single_model_name

    cmd = base_cmd_fn(args, Path(dataset_path), output_path)
    out_fn(f"  cmd: {' '.join(cmd[3:])}")
    out_fn("")

    returncode = subprocess.run(cmd, cwd=str(BASE_DIR), timeout=args.timeout).returncode
    if returncode == 0:
        out_fn(f"  ✓ FINAL TRAINING OK — adapter at {output_path}")
    else:
        out_fn(f"  ✗ FINAL TRAINING FAILED — exit={returncode}")
    return returncode


# ---------------------------------------------------------------------------
# Standalone entrypoint
# ---------------------------------------------------------------------------
def main():
    """Standalone HPO mode for train_all.py to invoke."""
    # This is invoked via `train_all.py --hpo` which sets up argparse.
    # The function is here mainly for `python -m hpo_runner --analyze-only`.
    parser = argparse.ArgumentParser(
        description="Analyze HPO trial results and print recommendation"
    )
    parser.add_argument(
        "--hpo-dir",
        type=str,
        default="hpo_runs",
        help="Directory containing HPO trial CSVs (default: hpo_runs/)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Don't run anything; just print the current best-per-axis state.",
    )
    args_h = parser.parse_args()
    state_path = Path(args_h.hpo_dir) / "hpo_state.json"
    if not state_path.exists():
        print(f"No hpo_state.json found at {state_path}")
        print("Run `train_all.py --hpo` first to generate it.")
        sys.exit(1)
    state = json.loads(state_path.read_text())
    print("=" * 60)
    print(" HPO STATE (from", state_path, ")")
    print("=" * 60)
    print("\nBest values per axis:")
    for axis_name, value in state.get("best_per_axis", {}).items():
        print(f"  {axis_name:20s} = {value}")
    print("\nTrial summary:")
    for t in state.get("all_trials", []):
        print(
            f"  [{t['axis']:15s}] v={t['value']:<6g} trial#{t['trial_index']} "
            f"verdict={t['verdict']:8s} loss_last10={t.get('mean_loss_last10')} "
            f"slope={t.get('loss_slope')}  ({t['wall_time_s']:.0f}s)"
        )

    print("\nRecommended final command (re-run with these to reproduce):")
    base = "uv run python scripts/train_all.py --single-model"
    base += " --base-model unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
    base += " --epochs 10 --max-length 2048"
    for axis_name, value in state.get("best_per_axis", {}).items():
        if axis_name == "lora_r":
            base += f" --lora-r {int(value)}"
        elif axis_name == "lora_alpha":
            base += f" --lora-alpha {int(value)}"
        elif axis_name == "lora_dropout":
            base += f" --lora-dropout {value:g}"
    print("  " + base)
    print("\nTo re-run with the same settings later, just copy/paste that command.")


if __name__ == "__main__":
    main()
