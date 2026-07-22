"""
hint_resample.py — Phase 2 engine for entropy-based hint resampling.

Flow for each (q, o_i):
  1. Feed completion through model token-by-token, compute Shannon entropy at each token.
  2. Identify .\n\n / ?\n\n tokens and their absolute entropy H_t.
  3. Retain positions where H_t > alpha → list L.
  4. Truncate at index floor(beta * len(L)) in L.
  5. Optionally insert hint, resample with model.
  6. Extract boxed answer, compare to ground truth, output metrics.
"""
import json
import re
import torch
import torch.nn.functional as F
from typing import Optional

# Matches tokens ending with .\n\n or ?\n\n
_SENTENCE_BREAK_PATTERN = re.compile(r"\.\\n\\n$|\\?\\n\\n$")


def _token_is_sentence_break(token_text: str) -> bool:
    """Check if a decoded token text ends with .\\n\\n or ?\\n\\n."""
    return token_text.endswith(".\n\n") or token_text.endswith("?\n\n")


def compute_entropy_sequence(
    model, tokenizer, input_ids: torch.Tensor, prompt_len: int
) -> list[dict]:
    """Feed the full sequence through the model once, record entropy at every position.

    Returns list of dicts: [{"abs_pos": int, "token_id": int, "token_text": str,
                              "entropy": float, "is_sentence_break": bool}, ...]
    Only includes completion tokens (positions >= prompt_len).
    """
    model.eval()
    device = input_ids.device
    results = []

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        logits = outputs.logits[0]  # [seq_len, vocab_size]

    seq_len = logits.size(0)
    for pos in range(prompt_len, seq_len):
        # logits at position pos predict the token at pos+1
        # So the entropy we record at position pos is from logits[pos-1]
        # which predicts the token at position pos
        if pos == 0:
            continue
        logit = logits[pos - 1].float()
        probs = F.softmax(logit, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        entropy = -(probs * log_probs).sum().item()

        token_id = input_ids[0, pos].item()
        token_text = tokenizer.decode([token_id], skip_special_tokens=True)
        is_break = _token_is_sentence_break(token_text)

        results.append({
            "abs_pos": pos,
            "rel_pos": pos - prompt_len,
            "token_id": token_id,
            "token_text": token_text,
            "entropy": entropy,
            "is_sentence_break": is_break,
        })

    return results


def extract_sentence_break_steps(entropy_seq: list[dict]) -> list[dict]:
    """Extract only the .\n\n and ?\n\n tokens from the entropy sequence.

    Returns list of dicts with keys: step (1-indexed), abs_pos, token_text, entropy.
    """
    steps = []
    step_counter = 0
    for item in entropy_seq:
        if item["is_sentence_break"]:
            step_counter += 1
            steps.append({
                "step": step_counter,
                "abs_pos": item["abs_pos"],
                "rel_pos": item["rel_pos"],
                "token_text": item["token_text"],
                "entropy": item["entropy"],
            })
    return steps


def compute_deltas(steps: list[dict]) -> list[float]:
    """Compute ΔH_t = H_t - H_{t-1} for each step.  First step has no delta (len = len(steps)-1)."""
    return [steps[i]["entropy"] - steps[i - 1]["entropy"] for i in range(1, len(steps))]


def apply_alpha_filter(steps: list[dict], alpha: float) -> list[int]:
    """Return indices into the original steps array where absolute entropy H > alpha."""
    qualifying = []
    for i, step in enumerate(steps):
        if step["entropy"] > alpha:
            qualifying.append(i)
    return qualifying


def choose_truncation_step(qualifying_indices: list[int], beta: float) -> Optional[dict]:
    """Choose the truncation step using beta.

    index = floor(beta * len(qualifying_indices))
    Returns the chosen index into steps, or None if no positions qualify.
    """
    if not qualifying_indices:
        return None
    idx = int(beta * len(qualifying_indices))
    idx = min(idx, len(qualifying_indices) - 1)
    return qualifying_indices[idx]


def locate_truncation_pos(
    generated_text: str, target_token_text: str, next_token_text: str
) -> Optional[int]:
    """Find the character position of a specific sentence-break token in generated_text.

    Uses the next token as disambiguation when there are multiple candidates.
    """
    candidates = []
    start = 0
    while True:
        pos = generated_text.find(target_token_text, start)
        if pos == -1:
            break
        candidates.append(pos)
        start = pos + 1

    for pos in candidates:
        after = generated_text[pos + len(target_token_text):]
        if after.startswith(next_token_text):
            return pos

    return candidates[0] if candidates else None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{...} in the text."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def run_single_resample(
    model,
    tokenizer,
    prompt_text: str,
    completion_text: str,
    ground_truth: str,
    steps: list[dict],
    alpha: float,
    beta: float,
    insert_hint: bool,
    hint_text: str,
    max_new_tokens: int,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> dict:
    """Run resample for a single (q, o_i) pair.

    Returns the metrics dict for this resample attempt.
    """
    device = next(model.parameters()).device

    # 1. Compute deltas for metrics; truncation candidates use absolute entropy.
    deltas = compute_deltas(steps)

    # 2. Alpha filter
    qualifying = apply_alpha_filter(steps, alpha)

    if not qualifying:
        # No positions qualify — return failure
        return {
            "is_success": 0,
            "chosen_truncation_index": None,
            "spike_delta_H": None,
            "spike_absolute_H": None,
            "resampled_H_sequence": [],
            "discarded_tokens_len": 0,
            "resampled_tokens_len": 0,
        }

    # 3. Beta selection
    chosen_step_idx = choose_truncation_step(qualifying, beta)
    chosen_step = steps[chosen_step_idx]
    chosen_delta = deltas[chosen_step_idx - 1] if chosen_step_idx > 0 else None

    # 4. Locate truncation position using pre-computed rel_pos (completion-relative)
    rel_pos = chosen_step["rel_pos"]
    # Tokenize just the completion text to find exact truncation character position
    comp_ids = tokenizer(completion_text, return_tensors="pt").input_ids[0]
    if rel_pos >= len(comp_ids):
        rel_pos = len(comp_ids) - 1
    # Decode completion tokens up to and including the chosen sentence-break token
    truncated_ids = comp_ids[: rel_pos + 1]
    text_before = completion_text  # fallback
    try:
        text_before = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    except Exception:
        pass
    discarded_ids = comp_ids[rel_pos + 1:]
    discarded_tokens_len = discarded_ids.size(0)

    # 5. Build resample prompt
    if insert_hint:
        resample_prompt = text_before + hint_text
    else:
        resample_prompt = text_before

    # 6. Generate
    inputs = tokenizer(resample_prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.size(1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_ids = outputs[0, prompt_len:]
    regenerated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    resampled_tokens_len = new_ids.size(0)

    # 7. Extract entropy sequence from new generation
    new_entropy_seq = compute_entropy_sequence(
        model, tokenizer, outputs[0], prompt_len
    )
    new_steps = extract_sentence_break_steps(new_entropy_seq)
    resampled_H_sequence = [s["entropy"] for s in new_steps]

    # 8. Check answer
    pred_answer = extract_boxed_answer(regenerated_text)
    is_success = 1 if (pred_answer is not None and pred_answer == str(ground_truth).strip()) else 0

    return {
        "is_success": is_success,
        "chosen_truncation_index": chosen_step["step"],
        "spike_delta_H": chosen_delta,
        "spike_absolute_H": chosen_step["entropy"],
        "resampled_H_sequence": resampled_H_sequence,
        "discarded_tokens_len": discarded_tokens_len,
        "resampled_tokens_len": resampled_tokens_len,
    }


def load_model_and_tokenizer(model_id: str, device_map: str = "auto"):
    """Load model and tokenizer. Separate function so caller can reuse across samples."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer
