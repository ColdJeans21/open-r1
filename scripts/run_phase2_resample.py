#!/usr/bin/env python
"""
Phase 2: Absolute-entropy based hint resampling experiment.

Reads Phase 1 JSON (zero-acc queries with completions), loads the model,
and for each (q, o_i) pair:
  1. Computes Shannon entropy at every token position
  2. Identifies .\n\n / ?\n\n step positions
  3. Applies alpha/beta filters to absolute entropy
  4. Truncates, optionally inserts hint, resamples
  5. Records success/failure and all metrics

Usage:
  python scripts/run_phase2_resample.py \
      --model_path /gpt/work/Ryan/xz/Qwen3-4B \
      --input_json ./zero_acc_with_entropy.json \
      --output_json ./resample_metrics.json \
      --alpha 1.5 --beta 0.5 --insert_hint \
      --hint_text "Wait, let me re-examine..." \
      --max_new_tokens 4096
"""
import argparse
import json
import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from open_r1.hint_resample import (
    load_model_and_tokenizer,
    compute_entropy_sequence,
    extract_sentence_break_steps,
    run_single_resample,
    extract_boxed_answer,
)


def main():
    parser = argparse.ArgumentParser(description="Phase 2: absolute-entropy hint resample")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--insert_hint", action="store_true", default=True)
    parser.add_argument("--no_insert_hint", dest="insert_hint", action="store_false")
    parser.add_argument("--hint_text", type=str,
                        default="Wait, I need to pause and carefully re-examine my reasoning above. Something might be off — let me go back through each step.\n\n")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device_map", type=str, default="auto")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device_map)
    device = next(model.parameters()).device

    # Load Phase 1 data
    print(f"Loading Phase 1 data from {args.input_json}...")
    with open(args.input_json) as f:
        phase1_data = json.load(f)

    if isinstance(phase1_data, dict):
        phase1_data = [phase1_data]
    print(f"Loaded {len(phase1_data)} entries")

    # Run resample for each entry
    results = []
    total_samples = 0
    total_success = 0

    for entry_idx, entry in enumerate(phase1_data):
        q_text = entry.get("q", "")
        ground_truth = str(entry.get("ground_truth", "")).strip()
        o_i_list = entry.get("o_i", [])

        print(f"\n--- Entry {entry_idx + 1}/{len(phase1_data)}: {len(o_i_list)} completions, GT={ground_truth} ---")

        for sample_idx, o_i_entry in enumerate(o_i_list):
            # Handle both old format (string) and new format (dict with entropy_steps)
            if isinstance(o_i_entry, str):
                comp_text = o_i_entry
                steps = []
                print(f"  [{entry_idx+1}:{sample_idx+1}] WARNING: old format, no pre-computed entropy — skipping entropy extraction")
                continue
            else:
                comp_text = o_i_entry.get("completion_text", "")
                steps = o_i_entry.get("entropy_steps", [])

            if not comp_text:
                continue
            total_samples += 1

            if len(steps) < 2:
                print(f"  [{entry_idx+1}:{sample_idx+1}] Only {len(steps)} sentence-break steps — skipping")
                results.append({
                    "question": q_text,
                    "sample_id": sample_idx + 1,
                    "is_success": 0,
                    "chosen_truncation_index": None,
                    "spike_delta_H": None,
                    "spike_absolute_H": None,
                    "resampled_H_sequence": [],
                    "discarded_tokens_len": 0,
                    "resampled_tokens_len": 0,
                })
                continue

            # Run single resample (steps already have entropy, abs_pos from Phase 1)
            metrics = run_single_resample(
                model=model,
                tokenizer=tokenizer,
                prompt_text="",  # not needed when abs_pos is in steps
                completion_text=comp_text,
                ground_truth=ground_truth,
                steps=steps,
                alpha=args.alpha,
                beta=args.beta,
                insert_hint=args.insert_hint,
                hint_text=args.hint_text,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            result = {
                "alpha_used": args.alpha,
                "beta_used": args.beta,
                "is_insert_hint": args.insert_hint,
                "question": q_text,
                "sample_id": sample_idx + 1,
                **metrics,
            }
            results.append(result)
            total_success += metrics["is_success"]

            status = "SUCCESS" if metrics["is_success"] else "FAIL"
            print(f"  [{entry_idx+1}:{sample_idx+1}] {status} | trunc_step={metrics['chosen_truncation_index']} | "
                  f"ΔH={metrics['spike_delta_H']:.4f}" if metrics['spike_delta_H'] is not None else f"  [{entry_idx+1}:{sample_idx+1}] {status} | no qualifying positions")

    # Save results
    with open(args.output_json, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"Phase 2 complete: {total_samples} samples, {total_success} successes "
          f"({100*total_success/max(1,total_samples):.1f}%)")
    print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
