"""
Input-normalization ablation @ epoch 900.

Runs the Kemeny Transformer (greedy/deterministic) on the jiggling / random /
repeat test datasets for two checkpoints that differ ONLY in whether the input
ranks were normalized during training:

  A) without_input_norm : args_700k_..._without_input_norm.conf (normalize_input=False)
  B) with_input_norm    : args_700k_..._mix_batch.conf          (normalize_input=True)

For each (dataset_type, n_voters, n_items) it writes
  <out>/<variant>/test_dataset_<type>/csv_results/
        test_dataset_<type>_nvoters_<v>_nitems_<i>_kemeny_distances.csv
with a per-sample `kemeny_distance` column (same format the plot scripts read).

Base ranking .npy files are (N, voters, items) with values = item->position rank
(verified: kemeny_dist(base, h1_ranking) reproduces the stored h1_distance).
"""

import os
import json
from time import time

import numpy as np
import pandas as pd
import torch

from kemeny_transformer.model.architecture import kemeny_transformer, EmbeddingType, DotDict
from kemeny_transformer.model.tokenization import KemenyTransformerTokenization
# Reuse the exact data-generation / training pipeline for scoring.
from kemeny_transformer.data.synthesis import kemeny_distance_batch, order_to_rank_batch


def clean_padded_permutations(permutation_tensor):
    """Padded permutation tensor -> list of numpy arrays with padding (0) removed.
    Identical to clean_padded_permutations in kemeny_transformer_training_ddp.py."""
    batch_list_np = []
    permutation_np_batch = permutation_tensor.cpu().detach().numpy()
    for i in range(permutation_np_batch.shape[0]):
        row = permutation_np_batch[i]
        batch_list_np.append(row[row != 0])
    return batch_list_np

OUT_ROOT = "results/input_norm_ablation_epoch_900"
SRC_FMT = "data/test/{t}/test_dataset_{t}_nvoters_{v}_nitems_{i}.npy"
DATASET_TYPES = ["jiggling", "random", "repeat"]
VOTERS = [6, 8, 10]
ITEMS = [200, 250]
BATCH = 128

VARIANTS = {
    "without_input_norm": {
        "config": "configs/args_700k_linear_various_voter_various_items_mix_batch_without_input_norm.conf",
        "checkpoint": ("outputs/checkpoints/various_input_kemeny_transformer/linear_embedding/"
                       "700k_various_voters_various_items_mix_batch_without_input_norm_3_encoder_2_decoder_"
                       "kemeny_transformer_without_guide/checkpoint_epoch_900.pkl"),
    },
    "with_input_norm": {
        "config": "configs/args_700k_linear_various_voter_various_items_mix_batch.conf",
        "checkpoint": ("outputs/checkpoints/various_input_kemeny_transformer/linear_embedding/"
                       "700k_various_voters_various_items_mix_batch_3_encoder_2_decoder_"
                       "kemeny_transformer_without_guide/checkpoint_epoch_900.pkl"),
    },
}


def build_model(cfg, checkpoint, device):
    args = DotDict()
    for k in ["dim_input", "dim_emb", "dim_ff", "numb_heads", "numb_layers_decoder",
              "numb_layers_encoder", "max_len_PE", "batchnorm"]:
        args[k] = cfg[k]
    args.conv_out_channels = cfg.get("conv_out_channels", 64)
    args.embedding_type = (EmbeddingType.LINEAR
                           if cfg.get("embedding_type", "linear").lower() == "linear"
                           else EmbeddingType.CONV)
    model = kemeny_transformer(
        embedding_type=args.embedding_type, input_dim=args.dim_input,
        embedding_dim=args.dim_emb, dim_ff=args.dim_ff, numb_heads=args.numb_heads,
        numb_layers_decoder=args.numb_layers_decoder,
        numb_layers_encoder=args.numb_layers_encoder, max_len_PE=args.max_len_PE,
        conv_out_channels=args.conv_out_channels, batchnorm=args.batchnorm,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    key = ("model_baseline_state_dict" if "model_baseline_state_dict" in ckpt
           else "model_train_state_dict")
    model.load_state_dict(ckpt[key])
    print(f"    loaded {key} (epoch {ckpt.get('epoch', -1)})")
    del ckpt
    model.eval()
    return model, args


def run_config(model, args, tokenizer, base, device):
    """base: (N, voters, items) item->position rank. returns per-sample kemeny distances.

    Mirrors the training/validation scoring path exactly:
      tokenize(raw base)  ->  model(deterministic=True)  ->  clean_padded_permutations
      ->  order_to_rank_batch  ->  kemeny_distance_batch(base, rankings).
    The base ranks are fed to the tokenizer untouched (no +1), as in
    kemeny_transformer_training_ddp.py.
    """
    N = base.shape[0]
    dists = np.zeros(N, dtype=np.float64)
    for start in range(0, N, BATCH):
        chunk = base[start:start + BATCH]
        instances = [chunk[b] for b in range(chunk.shape[0])]
        padded, pad_mask, voter_mask = tokenizer.tokenize(
            instances, embedding_type=args.embedding_type)
        padded = padded.to(device)
        pad_mask = pad_mask.to(device)
        voter_mask = voter_mask.to(device) if voter_mask is not None else None
        with torch.no_grad():
            orders, _, _ = model(padded, pad_mask, voter_mask=voter_mask, deterministic=True)
        cleaned = clean_padded_permutations(orders)
        rankings = order_to_rank_batch(cleaned)
        dists[start:start + chunk.shape[0]] = kemeny_distance_batch(instances, rankings)
    return dists


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    summary_rows = []
    for variant, spec in VARIANTS.items():
        print(f"\n=== variant: {variant} ===")
        cfg = json.load(open(spec["config"]))
        model, args = build_model(cfg, spec["checkpoint"], device)
        normalize_input = cfg.get("normalize_input", "False") == "True"
        tokenizer = KemenyTransformerTokenization(
            max_voters=args.dim_input, pad_value=0.0, normalize_input=normalize_input)
        print(f"    normalize_input = {normalize_input}")

        for t in DATASET_TYPES:
            out_dir = os.path.join(OUT_ROOT, variant, f"test_dataset_{t}", "csv_results")
            os.makedirs(out_dir, exist_ok=True)
            for v in VOTERS:
                for i in ITEMS:
                    src = SRC_FMT.format(t=t, v=v, i=i)
                    if not os.path.exists(src):
                        print(f"    [skip] missing {src}")
                        continue
                    base = np.load(src, allow_pickle=True).astype(np.int64)
                    t0 = time()
                    dists = run_config(model, args, tokenizer, base, device)
                    elapsed = time() - t0
                    df = pd.DataFrame({"sample_id": np.arange(len(dists)),
                                       "kemeny_distance": dists})
                    out_csv = os.path.join(
                        out_dir, f"test_dataset_{t}_nvoters_{v}_nitems_{i}_kemeny_distances.csv")
                    df.to_csv(out_csv, index=False)
                    mean_d = float(dists.mean())
                    summary_rows.append({"variant": variant, "dataset": t,
                                         "n_voters": v, "n_items": i,
                                         "mean_kemeny": mean_d, "n_samples": len(dists),
                                         "time_sec": elapsed})
                    print(f"    {t} v{v} i{i}: mean_kemeny={mean_d:8.2f}  "
                          f"({len(dists)} samples, {elapsed:.1f}s)")
        del model
        torch.cuda.empty_cache()

    os.makedirs(OUT_ROOT, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT_ROOT, "summary_means.csv"), index=False)
    print(f"\nWrote summary -> {os.path.join(OUT_ROOT, 'summary_means.csv')}")


if __name__ == "__main__":
    main()
