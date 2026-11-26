"""Clipped token-ratio updater for coupled distribution credits, not stock GRPO.

Advantages and initial-policy KL credits are frozen at each rollout snapshot.
The first-pass gradient matches the sequence REINFORCE estimator. Later passes
are a local surrogate, not an unbiased off-policy estimator of batch-level KL.
No reward-std or per-response-length normalization is used.
"""

import contextlib

import numpy as np
import torch
import torch.distributed as dist
from ddp_passage_rl import gather_objects, prompt_loo_baselines

from stochastic_llm.rl import completion_token_log_probs


def validate_update_config(cfg):
    method = cfg.get("policy_update", "reinforce")
    if method not in ("reinforce", "grpo"):
        raise ValueError("policy_update must be reinforce or grpo")
    if cfg.get("quality_baseline", "global-loo") not in ("global-loo", "prompt-loo"):
        raise ValueError("Unknown quality baseline")
    if method == "grpo":
        passes = cfg.get("grpo_passes", 2)
        epsilon = cfg.get("grpo_clip_epsilon", 0.2)
        if isinstance(passes, bool) or not isinstance(passes, int) or not 1 <= passes <= 4:
            raise ValueError("grpo_passes must be an integer between one and four")
        if not np.isfinite(epsilon) or not 0 < epsilon < 1:
            raise ValueError("GRPO clip epsilon must be finite and between zero and one")
        if cfg.get("advantage_normalization", "none") != "none":
            raise ValueError("Coupled credits must not be re-centered or std-normalized")
        if cfg.get("grpo_loss_normalization", "sequence-sum") != "sequence-sum":
            raise ValueError("Use summed token losses per answer, not answer-length normalization")
        if cfg.get("grpo_reference_penalty", "frozen-sequence-credit") != "frozen-sequence-credit":
            raise ValueError("Unknown GRPO reference-policy penalty")
    return method


def token_surrogate(current, old, task_credit, kl_credit, beta, epsilon):
    """Return per-token loss, ratio and task-clipping indicators.

    The reference term is separate and unclipped. Its coefficient is the frozen
    (possibly prompt-centered) sequence log ratio to the initial actor. This
    preserves the existing first-pass gradient, rather than substituting a
    differently scaled tokenwise KL estimator.
    """
    if not np.isfinite(beta) or beta < 0 or not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("Invalid reference coefficient or clipping epsilon")
    if not (current.shape == old.shape == task_credit.shape == kl_credit.shape):
        raise ValueError("Need one snapshot and detached credit per completion token")
    if any(not torch.isfinite(v).all() for v in (current, old, task_credit, kl_credit)):
        raise FloatingPointError("Nonfinite token probabilities or policy credits")
    ratio = torch.exp(current - old.detach())
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Nonfinite importance ratio; refusing a stale rollout update")
    advantage = task_credit.detach()
    direct = ratio * advantage
    clipped = ratio.clamp(1 - epsilon, 1 + epsilon) * advantage
    loss = -torch.minimum(direct, clipped) + beta * ratio * kl_credit.detach()
    active = ((advantage > 0) & (ratio > 1 + epsilon)) | ((advantage < 0) & (ratio < 1 - epsilon))
    return loss, ratio, active


def distributed_grpo_update(
    policy,
    optimizer,
    rows,
    credit,
    pad_id,
    microbatch,
    reference_beta,
    *,
    passes=2,
    clip_epsilon=0.2,
    reference_kl_centering="none",
    loss_divisor=1.0,
    record_log_probs=False,
    scheduler=None,
):
    """Reuse ONE globally credited rollout batch, with one DDP update per pass.

    Snapshot all old/reference probabilities before stepping either optimizer.
    DDP averages ranks; divide summed token losses by local answer count so
    token-summed sequence weighting matches the existing REINFORCE update.
    The scheduler advances once per optimizer pass, not once per rollout batch.
    """
    validate_update_config(
        {"policy_update": "grpo", "grpo_passes": passes, "grpo_clip_epsilon": clip_epsilon}
    )
    if reference_kl_centering not in ("none", "prompt-loo"):
        raise ValueError("Unknown initial-policy KL centering mode")
    if not np.isfinite(loss_divisor) or loss_divisor <= 0:
        raise ValueError("Loss divisor must be fixed, finite, and positive")
    if not rows or microbatch < 1 or len(credit) != len(rows):
        raise ValueError("Need nonempty rows, positive microbatch and one credit per answer")
    sizes = gather_objects(len(rows))
    if len(set(sizes)) != 1:
        raise ValueError("DDP ranks need equal local answer counts")
    actor = policy.module.actor
    actor.eval()
    old, reference = [], []
    for start in range(0, len(rows), microbatch):
        chunk = rows[start : start + microbatch]
        lengths = [len(r["completion_ids"]) for r in chunk]
        with torch.no_grad():
            values = completion_token_log_probs(actor, chunk, pad_id)
            old.extend(t.detach().cpu().clone() for t in values.split(lengths))
        with torch.no_grad(), actor.disable_adapter():
            values = completion_token_log_probs(actor, chunk, pad_id)
            reference.extend(t.detach().cpu().clone() for t in values.split(lengths))
    current_values = np.array([t.sum().item() for t in old])
    reference_values = np.array([t.sum().item() for t in reference])
    deltas = current_values - reference_values
    baselines = np.zeros(len(rows))
    if reference_kl_centering == "prompt-loo":
        parts = gather_objects(
            [{"case_id": r["case_id"], "delta": float(d)} for r, d in zip(rows, deltas)]
        )
        flat = [r for part in parts for r in part]
        all_baselines = prompt_loo_baselines(
            [r["delta"] for r in flat], [r["case_id"] for r in flat]
        )
        offset = sum(len(part) for part in parts[: dist.get_rank()])
        baselines = all_baselines[offset : offset + len(rows)]
    centered = deltas - baselines
    pass_records = []
    for iteration in range(passes):
        optimizer.zero_grad(set_to_none=True)
        loss_sum, ratio_sum, squared_log_ratio_sum, clipped_sum, tokens = 0.0, 0.0, 0.0, 0.0, 0
        for start in range(0, len(rows), microbatch):
            chunk = rows[start : start + microbatch]
            lengths = torch.tensor([len(r["completion_ids"]) for r in chunk], device=credit.device)
            old_chunk = torch.cat(old[start : start + len(chunk)]).to(credit.device)
            task = torch.repeat_interleave(credit[start : start + len(chunk)].detach(), lengths)
            kl = torch.repeat_interleave(
                torch.tensor(
                    centered[start : start + len(chunk)], device=credit.device, dtype=credit.dtype
                ),
                lengths,
            )
            sync = contextlib.nullcontext() if start + microbatch >= len(rows) else policy.no_sync()
            with sync:
                current = policy(chunk, pad_id, token_level=True)
                losses, ratio, active = token_surrogate(
                    current, old_chunk, task, kl, reference_beta, clip_epsilon
                )
                loss = losses.sum() / (len(rows) * loss_divisor)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite GRPO loss")
                loss.backward()
            loss_sum += loss.detach().item()
            ratio_sum += ratio.detach().sum().item()
            squared_log_ratio_sum += (current.detach() - old_chunk).square().sum().item()
            clipped_sum += active.sum().item()
            tokens += len(current)
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in actor.parameters() if p.requires_grad], 1.0
        )
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite GRPO gradient")
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        stats = torch.tensor(
            [loss_sum, ratio_sum, squared_log_ratio_sum, clipped_sum, tokens],
            device=credit.device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats)
        pass_records.append(
            {
                "pass": iteration + 1,
                "policy_gradient_loss": stats[0].item() / dist.get_world_size(),
                "token_ratio_mean": stats[1].item() / stats[4].item(),
                "old_policy_log_ratio_rms": (stats[2] / stats[4]).sqrt().item(),
                "task_clip_fraction": stats[3].item() / stats[4].item(),
                "gradient_norm": float(norm),
            }
        )
    stats = torch.tensor(
        [deltas.mean(), np.mean(deltas**2), np.mean(centered**2)],
        device=credit.device,
        dtype=torch.float64,
    )
    dist.all_reduce(stats)
    stats /= dist.get_world_size()
    result = {
        "policy_gradient_loss": float(np.mean([r["policy_gradient_loss"] for r in pass_records])),
        "gradient_norm": max(r["gradient_norm"] for r in pass_records),
        "reference_sequence_kl_estimate": stats[0].item(),
        "reference_kl_credit_rms_raw": stats[1].sqrt().item(),
        "reference_kl_credit_rms_centered": stats[2].sqrt().item(),
        "objective_loss_divisor": float(loss_divisor),
        "optimizer_steps_this_batch": passes,
        "grpo_clip_fraction": float(np.mean([r["task_clip_fraction"] for r in pass_records])),
        "grpo_last_token_ratio_mean": pass_records[-1]["token_ratio_mean"],
        "grpo_last_old_policy_log_ratio_rms": pass_records[-1]["old_policy_log_ratio_rms"],
    }
    for record in pass_records:
        result.update(
            {f"grpo_pass_{record['pass']}_{k}": v for k, v in record.items() if k != "pass"}
        )
    if record_log_probs:
        result["_row_metrics"] = [
            {
                "policy_logprob_before_update": float(cur),
                "initial_policy_logprob": float(ref),
                "reference_log_ratio": float(delta),
                "reference_loo_baseline": float(baseline),
                "centered_reference_log_ratio": float(center),
                "grpo_old_token_logprobs": old[i].tolist(),
            }
            for i, (cur, ref, delta, baseline, center) in enumerate(
                zip(current_values, reference_values, deltas, baselines, centered)
            )
        ]
    return result
