# RL Post-Training Recipe

> **Methodology credit**: This document implements techniques from
> "MAI-Thinking-1: Building a Hill-Climbing Machine" by The Microsoft AI Team
> (June 2026, 109 pages). The specific methodology we adopted is described in
> [section 3.1.1, 3.1.2, and 3.4.2 of that paper](paper-citation). We thank the Microsoft AI Team
> for sharing their development methodology in detail.
> 
> **Section reference**: MAI-Thinking-1 §3.1.1, §3.1.2, §3.4.2 — Reinforcement Learning Recipe
> **What we took**: Adaptive entropy control for stability, a dynamic length penalty scaled by problem pass rate, and the use of coarse-grained reward graders.
> **What we adapted**: We target a 3B-14B model scale on consumer hardware (RTX 4080), replacing the frontier-scale MoE architecture with dense/LoRA-based post-training.
> **What we did NOT take**: 8K GB200 infrastructure, MoE architecture, the YOLO training framework, or the 35B model weights.
>
> *If the paper later gets a public URL, replace `(paper-citation)` in this block
> with the real URL. The section number + title is the canonical link for now.*

## 1. Why this doc exists

AttackLM is currently an SFT-only (Supervised Fine-Tuning) platform. While SFT provides a strong foundation for security-domain knowledge, it does not teach the model to "reason" through complex attack chains or self-correct.

This document serves as the official implementation recipe for the day AttackLM adds Reinforcement Learning (RL). It is a forward-looking specification, not an active implementation. When the RL stack is initiated, the implementing engineer must follow these constraints to ensure training stability and avoid the common pitfalls of RL (e.g., reward hacking, numeric collapse).

## 2. GRPO Loss Baseline

We adopt **Group Relative Policy Optimization (GRPO)** as the primary RL algorithm. Unlike PPO, which requires a separate value-function (critic) model, GRPO computes the baseline from the average reward of a group of sampled completions for the same prompt.

- **Baseline**: Yu et al. 2025 token-level policy gradient.
- **Implementation**: Use `trl.GRPOTrainer` (HuggingFace TRL) for the training loop.
- **Objective**: Maximize the relative reward of a completion within its group while minimizing the KL divergence from the reference (SFT) model.

## 3. Adaptive Entropy Control (§3.1.1)

Standard entropy bonuses often underperform because a fixed coefficient either fails to prevent premature convergence or biases the gradient too heavily. We implement an **integral controller** to track a target entropy.

### The Controller
The goal is to maintain a target entropy $H^\star = 0.3$. We adjust the entropy coefficient $k$ dynamically based on the measured entropy $\hat{H}(\pi_\theta)$ of the current policy.

**Update Rule:**
$$k \leftarrow \text{clip}(k + \delta \cdot \text{sign}(H^\star - \hat{H}(\pi_\theta)), 0, k_{max})$$

- **Target Entropy ($H^\star$):** 0.3
- **Upper Clip Bound ($k_{max}$):** 2.5
- **Step Size ($\delta$):** 0.25

**Reasoning**: This controller prevents "entropy collapse" (where the model becomes overly confident in a few patterns) without biasing the gradient toward high-entropy, low-quality noise.

## 4. Outer Ratio Clip (§3.1.1)

To prevent catastrophic policy shifts during a single update step, we implement an **Outer Ratio Clip**, a variant of the dual-clip PPO mechanism.

- **$r_{max}$**: 50
- **$r_{min}$**: 0

If the ratio between the current policy and the reference policy exceeds these bounds, the gradient is clipped. This ensures that no single high-reward sample can pivot the model's weights too aggressively, which is critical for stability in security domains where a single "perfect" attack string might have an extreme reward.

## 5. Length Penalty scaled by Problem Pass Rate (§3.1.2)

RL models often "hack" length-based rewards by producing verbose, repetitive chains of thought. We implement a dynamic length penalty that scales based on the difficulty of the problem.

**Formula:**
$$R_{len}(y) = \rho_q \cdot \frac{|y|}{\ell_{max}}$$

- $|y|$: Length of the generated response.
- $\ell_{max}$: Maximum allowable sequence length.
- $\rho_q$: The **pass rate** of problem $q$ (the fraction of samples in the group that were correct).

**Logic:**
- **Easy Problems (High $\rho_q$):** Stronger penalty. If most samples are correct, the model is encouraged to be concise.
- **Hard Problems (Low $\rho_q$):** Weaker penalty. For complex attacks, the model is given "room to think" without being penalized for the length of its reasoning chain.

## 6. Self-Distillation for Crash Recovery (§3.1.4)

RL training is prone to numeric collapse or "reward hacking" where the model finds a loophole in the grader. When this occurs, we use **Self-Distillation**.

**Process:**
1. Identify the "best" rollout traces from the buffer (highest reward, verified correctness).
2. Perform a short SFT run (re-SFT) on these high-quality traces.
3. Reset the RL optimizer but keep the re-SFT weights.

**Use Cases:**
- **Numeric Collapse**: Recovering from $\text{NaN}$ losses or gradient explosions.
- **Base Policy Migration**: Carrying over progress when moving to a new base model.
- **Format Updates**: Quickly updating the chat template without losing RL progress.
- **Reward-Hack Filtering**: Using the SFT phase to "wash" the buffer of reward-hacking samples.

## 7. Coarse vs Granular Graders (§3.4.2 + App E)

The choice of reward granularity is critical. We adopt **Coarse Graders** (binary 0/1 or ternary 0/1/2) rather than granular, continuous scores.

- **Findings**: Coarse graders outperform granular graders because the model cannot "hack" the reward by slightly adjusting token probabilities to trick a regression-based judge.
- **Application**: Any reward model built for AttackLM (e.g., a judge checking if a payload is valid) must output a discrete, coarse score.

## 8. Top-p Mask Reuse (§3.1.3)

To optimize compute and improve stability, we reuse the truncation masks from the rollout phase.

- **Mechanism**: During the RL rollout, the model samples with $p=0.97$. We save the mask of the top-p tokens.
- **Application**: During the training update, we apply the same mask to the loss calculation.
- **Benefit**: Prevents the model from being penalized for tokens that were never likely to be sampled, focusing the gradient on the active policy subspace.

## 9. What we DON'T adopt

To keep AttackLM accessible to the research community, we explicitly exclude the following frontier-scale requirements:
- **Hardware**: No 8K GB200 cluster. We target single-node RTX 4080/A100 setups.
- **Architecture**: No Mixture-of-Experts (MoE). We stick to dense Transformer architectures (Qwen2.5-Coder) for ease of fine-tuning.
- **Framework**: No "YOLO" distributed framework. We use HF TRL, Axolotl, and standard PyTorch.
- **Model**: We do not use the 35B MAI-Thinking-1 weights; we use the recipe to train 3B-14B models.

## 10. Open questions for the day we build it

- **HPO of $H^\star$ and $k_{max}$**: Does the target entropy of 0.3 scale linearly with model size (e.g., do we need $H^\star = 0.2$ for a 14B model)?
- **Scale-dependent stability**: Does the adaptive entropy controller provide a measurable gain at the 3B scale, or is it only necessary for frontier-scale models (35B+)?
- **Reward Model Source**: Should we use a public judge (GPT-4o) for coarse grading, or implement a "self-judge" based on the a la-carte evaluation suite in `attacklm-dataset`?
- **Length Penalty $\rho_q$**: We currently lack a per-problem pass rate for our SFT la-carte suite. Can we derive $\rho_q$ from a held-out NLL distribution, or must we generate 10+ samples per prompt to compute it?
