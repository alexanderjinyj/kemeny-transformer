# Kemeny Transformer

A transformer-based neural network for solving the **Kemeny consensus ranking** problem. Given a set of rankings (votes) from multiple voters over a set of candidates, the model learns to produce an aggregate ranking that minimizes the total Kemeny distance (number of pairwise disagreements) to all input rankings.

## Problem Definition

The **Kemeny ranking problem** is: given `n` voters each providing a complete ranking over `m` candidates, find the ranking that minimizes the sum of Kendall tau distances to all voter rankings. This is NP-hard, so exact solutions (via Gurobi ILP solver) are slow for large inputs. This project trains a transformer to approximate the optimal solution in near-constant time.

**Input**: A matrix of shape `(num_voters, num_candidates)` where entry `[i, j]` is the rank position that voter `i` assigns to candidate `j` (0-indexed).

**Output**: A permutation over candidates representing the consensus ranking.

## Project Structure

```
kemeny-transformer/
    kemeny_transformer/              # Python package
        model/
            architecture.py          # Transformer encoder-decoder model
            tokenization.py          # Converts rankings to padded tensors
        utils/
            graph.py                 # Preference graph construction
            kemeny_distance.py       # Kemeny distance computation (numba JIT)
            gurobi_solver.py         # Exact ILP solver (Gurobi)
        data/
            synthesis.py             # Synthetic ranking data generation
        baselines/
            kwik_sort.py             # KwikSort heuristic
            markov_chain.py          # Markov chain aggregation
            heuristic_ranking.py     # Beck-Lin heuristic (1983)
        evaluation/
            metrics.py               # Fairness metrics
    scripts/
        train_ddp.py                 # Training (single/multi-GPU with DDP)
        inference.py                 # Inference with greedy/beam search
        run_gurobi_test.py           # Compute exact Gurobi solutions
        run_result_analysis.py       # Analyze transformer vs Gurobi results
        generate_test_dataset.py     # Generate test datasets
        generate_scalarity_dataset.py # Generate scalability test datasets
        generate_validation_dataset.py # Generate validation datasets
        test_various_input.py        # Test on variable-size inputs
        plot_results.py              # Visualization
    configs/                         # Training configurations
    notebooks/                       # Jupyter analysis notebooks
    r_scripts/                       # R statistical analysis
```

---

## 1. Data Generation

The model is trained on **synthetically generated** ranking data. No pre-existing dataset is stored — data is generated on-the-fly during training.

### How Rankings Are Generated

Each training sample is a set of `num_voters` rankings over `num_candidates` items. The generation process creates rankings with controllable levels of agreement/disagreement between voters using three perturbation methods:

#### Method 1: Random (`random`)
Each voter's ranking is an independent random permutation. This produces maximum disagreement — there is little consensus.

```
Voter 1: [3, 0, 2, 1, 4]   # completely random
Voter 2: [1, 4, 0, 3, 2]   # completely random (independent)
```

#### Method 2: Repeat (`repeat`)
A "ground truth" ranking is first generated. Each subsequent voter either copies a previous voter's ranking (with probability `phi`) or generates a fresh random permutation. Higher `phi` = more agreement between voters.

```
Ground truth: [0, 1, 2, 3, 4]
Voter 1: [0, 1, 2, 3, 4]   # copies ground truth
Voter 2: [0, 1, 2, 3, 4]   # copies voter 1 (with prob phi)
Voter 3: [2, 4, 0, 1, 3]   # fresh random (with prob 1-phi)
```

#### Method 3: Jiggling (`jiggling`)
Starting from a base ranking, each voter's ranking is created by "jiggling" — randomly swapping each candidate with another candidate at an exponentially-distributed distance. Nearby swaps are more likely than distant ones, creating smooth perturbations.

```
Base:    [0, 1, 2, 3, 4]
Voter 1: [0, 2, 1, 3, 4]   # small perturbation (swap positions 1,2)
Voter 2: [0, 1, 3, 2, 4]   # small perturbation (swap positions 2,3)
```

#### Mix Batch (for fine-tuning)
During fine-tuning, the `fine_tuning_mix` method creates each batch by mixing all three perturbation types. Within each batch, some samples use random, some use repeat, and some use jiggling.

### Controlling Data Difficulty

The `phi_range` parameter in the config controls the perturbation strength:
- `phi = 0.0`: Maximum noise (pure random permutations, no consensus)
- `phi = 1.0`: Minimum noise (voters nearly identical, easy consensus)
- Training uses a range like `[0.0, 0.1, ..., 1.0]`, randomly sampling `phi` per batch

### Variable Input Sizes

The model supports variable numbers of voters and candidates per sample:
- `num_voters_range: [6, 10]` — each batch randomly samples a voter count in this range
- `num_items_range: [90, 110]` — each batch randomly samples a candidate count in this range
- The tokenizer pads shorter sequences and creates attention masks so the transformer handles variable sizes

### Generating Test/Validation Datasets

Test and validation datasets are pre-generated and saved as `.npy` files:

```bash
# Generate test datasets (25, 50, 75, 100 candidates, 8 voters)
PYTHONPATH=. python scripts/generate_test_dataset.py

# Generate scalability test datasets (20, 50, 100, 150, 200 candidates)
PYTHONPATH=. python scripts/generate_scalarity_dataset.py

# Generate validation datasets (6 and 10 voters, 100 candidates)
PYTHONPATH=. python scripts/generate_validation_dataset.py
```

Each script generates datasets for all three perturbation types (random, repeat, jiggling) and saves them under `test_dataset/` or `validate_dataset/`.

---

## 2. Model Architecture

The model is a standard **encoder-decoder transformer** adapted for the ranking problem:

### Tokenization (Preprocessing)

Raw rankings of shape `(num_voters, num_candidates)` are transposed to `(num_candidates, num_voters)` — each candidate becomes a "token" whose features are its rank positions across all voters. Optionally, ranks are normalized to `[0, 1]` by dividing by `(num_candidates - 1)`.

For variable-size inputs:
- **Voter dimension**: Padded to `max_voters` (for linear embedding) or handled natively (for conv embedding)
- **Candidate dimension**: Padded to `max_candidates` in the batch
- **Attention masks**: Created so the model ignores padded positions

### Embedding Layer

Two options:
- **Linear**: A simple `nn.Linear(max_voters, dim_emb)` — requires fixed voter count
- **Conv**: 1D convolutions over the voter dimension, followed by pooling — handles variable voter counts natively

### Encoder

A standard multi-layer transformer encoder with multi-head self-attention. Each candidate attends to every other candidate, producing contextualized embeddings that capture inter-candidate preference patterns.

### Decoder (Autoregressive)

The decoder builds the consensus ranking one candidate at a time:
1. Start with a learned `start_placeholder` token
2. At each step `t`:
   - Self-attention over all previously selected candidates
   - Cross-attention over encoder outputs (masking already-selected and padded candidates)
   - Output attention weights as probabilities over remaining candidates
   - Select next candidate (argmax for greedy, sampling for training/beam search)
   - Feed the selected candidate's encoder embedding (+ positional encoding) as input for step `t+1`
3. Repeat until all candidates are ranked

### Training Objective (REINFORCE)

The model is trained with **policy gradient (REINFORCE)** — not supervised learning. There are no ground-truth labels. Instead:

1. **Actor (model_train)**: Samples a ranking by running the decoder with stochastic sampling (`deterministic=False`)
2. **Baseline (model_baseline)**: Produces a greedy ranking (`deterministic=True`) — this is a separate copy of the model updated periodically
3. **Reward signal**: The Kemeny distance of each ranking to the input voter rankings
4. **Advantage**: `kemeny_dist_train - kemeny_dist_baseline` (how much worse the sampled ranking is vs the baseline)
5. **Loss**: `mean(advantage * log_prob)` — REINFORCE policy gradient

The baseline model is updated when the training model consistently outperforms it on a rollout evaluation batch. This creates a self-improving loop where the model continually raises its own bar.

---

## 3. Training

### Configuration

Training is controlled by a JSON config file. Key parameters:

| Parameter | Description | Example |
|---|---|---|
| `dim_emb` | Embedding/hidden dimension | `128` |
| `dim_ff` | Feed-forward dimension | `512` |
| `numb_heads` | Attention heads | `8` |
| `numb_layers_encoder` | Encoder layers | `2` |
| `numb_layers_decoder` | Decoder layers | `3` |
| `embedding_type` | `"linear"` or `"conv"` | `"linear"` |
| `lr` | Learning rate | `1e-4` |
| `nb_epochs` | Number of epochs | `1000` |
| `nb_batch_per_epoch` | Batches per epoch | `250` |
| `bsz` | Batch size | `512` |
| `num_voters_range` | Min/max voters | `[6, 10]` |
| `num_items_range` | Min/max candidates | `[90, 110]` |
| `phi_range` | Perturbation strengths | `[0.0, ..., 1.0]` |
| `data_generation_method` | How to generate data | `"random"` / `"fine_tuning_mix"` |
| `normalize_input` | Normalize ranks to [0,1] | `"True"` |
| `baseline_alpha` | Baseline update threshold | `0.05` |
| `checkpoint_file` | Resume from checkpoint | `"checkpoint_epoch_400.pkl"` or `null` |

### Running Training

**Single GPU:**
```bash
PYTHONPATH=. python scripts/train_ddp.py configs/args_8_voter_100_items.conf
```

**Multi-GPU (DDP):**
```bash
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_ddp.py configs/args_8_voter_100_items.conf
```

### Two-Phase Training Strategy

1. **Phase 1 — Pre-training** (`data_generation_method: "random"`):
   Train on random rankings with fixed voter/candidate counts (e.g., 8 voters, 100 candidates). This teaches the model the basic structure of the ranking problem.

2. **Phase 2 — Fine-tuning** (`data_generation_method: "fine_tuning_mix"`):
   Fine-tune on mixed perturbation types with variable voter/candidate counts. Set `checkpoint_file` to the Phase 1 checkpoint. This improves generalization to different input sizes and perturbation patterns.

### What Happens During Training

Each epoch:
1. Generate `nb_batch_per_epoch` batches of synthetic rankings on-the-fly
2. For each batch:
   - Actor samples a ranking (stochastic)
   - Baseline produces a greedy ranking (deterministic)
   - Compute Kemeny distances for both
   - Compute advantage and REINFORCE loss
   - Backpropagate
3. Run validation on pre-generated validation datasets
4. Rollout evaluation: compare actor vs baseline on fresh data, update baseline if actor wins
5. Save checkpoint every `save_every` epochs

### Outputs

- **Checkpoints**: `kemeny_transformer_checkpoint/.../*.pkl` (model weights, optimizer state, epoch)
- **Validation logs**: `validate_logs/.../validation_*.txt` (per-epoch gap metrics)
- **Rollout logs**: `roll_out_logs/.../roll_out_*.txt` (actor vs baseline performance)

---

## 4. Inference

After training, run inference to produce consensus rankings on test datasets:

```bash
# Both greedy and beam search:
PYTHONPATH=. python scripts/inference.py \
    --config configs/args_8_voter_100_items.conf \
    --checkpoint kemeny_transformer_checkpoint/.../checkpoint_epoch_500.pkl \
    --dataset test_dataset/test_dataset_random.npy \
    --output-dir test_dataset/transformer_test_result/test_data \
    --output-name test_dataset_random

# Greedy only (faster):
PYTHONPATH=. python scripts/inference.py \
    --config configs/args_8_voter_100_items.conf \
    --checkpoint path/to/checkpoint.pkl \
    --dataset test_dataset/test_dataset_random.npy \
    --output-dir results/ \
    --output-name test_dataset_random \
    --mode greedy

# Beam search with custom beam size:
PYTHONPATH=. python scripts/inference.py \
    --config configs/args_8_voter_100_items.conf \
    --checkpoint path/to/checkpoint.pkl \
    --dataset test_dataset/test_dataset_random.npy \
    --output-dir results/ \
    --output-name test_dataset_random \
    --mode beam_search \
    --beam-size 10
```

### Decoding Strategies

- **Greedy**: At each step, pick the candidate with the highest probability. Fast, deterministic.
- **Beam search**: Run the model `beam_size` times with stochastic sampling, then pick the ranking with the lowest Kemeny distance. Slower but produces better results.

### Output Format

The output `.npy` file contains a dictionary with:
- `final ranking greedy {name} permutation` — greedy rankings `(num_samples, num_candidates)`
- `greedy time {name}` — total greedy inference time
- `greedy kemeny distance {name}` — per-sample Kemeny distances
- `greedy kemeny distance mean {name}` — mean Kemeny distance
- `final ranking beam search {name} permutation` — beam search rankings `(num_samples, beam_size, num_candidates)`
- `beam search time {name}` — total beam search time
- `beam search kemeny distance {name}` — per-sample best Kemeny distances
- `beam search kemeny distance mean {name}` — mean best Kemeny distance
- `beam search optimal rankings {name}` — best ranking per sample across beams

---

## 5. Evaluation Against Baselines

### Computing Gurobi Optimal Solutions

Gurobi solves the Kemeny problem exactly via integer linear programming (ILP). It's the ground truth but very slow for large inputs.

```bash
# Standard test (100 candidates):
PYTHONPATH=. python scripts/run_gurobi_test.py --mode standard --data-types random repeat jiggling

# Ablation (variable candidates):
PYTHONPATH=. python scripts/run_gurobi_test.py --mode ablation --data-types random --num-candidates 25 50

# Scalability test:
PYTHONPATH=. python scripts/run_gurobi_test.py --mode scalarity --data-types repeat --num-candidates 150 --max-samples 25
```

### Analyzing Results

Compare transformer output against Gurobi optimal:

```bash
# Standard analysis:
PYTHONPATH=. python scripts/run_result_analysis.py --data-type random

# Fine-tuning results:
PYTHONPATH=. python scripts/run_result_analysis.py --data-type random --fine-tuning

# Scalability analysis:
PYTHONPATH=. python scripts/run_result_analysis.py --scalarity --data-types random repeat jiggling --num-candidates 20 50 100 150
```

### Other Baselines

The project includes three heuristic baselines for comparison:
- **KwikSort** (`kemeny_transformer.baselines.kwik_sort`): Randomized quicksort on the tournament graph
- **Markov Chain** (`kemeny_transformer.baselines.markov_chain`): Stationary distribution of a preference-based Markov chain
- **Beck-Lin Heuristic** (`kemeny_transformer.baselines.heuristic_ranking`): Greedy maximize-agreement / minimize-regret algorithms

---

## 6. End-to-End Workflow Example

```bash
# 1. Generate validation data
PYTHONPATH=. python scripts/generate_validation_dataset.py

# 2. Train (Phase 1: random, fixed size)
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_ddp.py configs/args_8_voter_100_items.conf

# 3. Train (Phase 2: fine-tune with mixed data, variable size)
#    Edit config: set checkpoint_file, change data_generation_method to "fine_tuning_mix"
PYTHONPATH=. torchrun --nproc_per_node=2 scripts/train_ddp.py configs/args_700k_linear_various_voter_various_items_mix_batch.conf

# 4. Generate test datasets
PYTHONPATH=. python scripts/generate_test_dataset.py

# 5. Run inference
PYTHONPATH=. python scripts/inference.py \
    --config configs/args_700k_linear_various_voter_various_items_mix_batch.conf \
    --checkpoint kemeny_transformer_checkpoint/.../checkpoint_epoch_650.pkl \
    --dataset test_dataset/ablation_test_dataset/test_dataset_random_100.npy \
    --output-dir test_dataset/transformer_test_result/test_data \
    --output-name test_dataset_random

# 6. Compute Gurobi baselines
PYTHONPATH=. python scripts/run_gurobi_test.py --mode standard --data-types random

# 7. Analyze results
PYTHONPATH=. python scripts/run_result_analysis.py --data-type random
```

## Environment Setup

The project uses the `kemeny_transformer` conda environment:

```bash
conda activate kemeny_transformer
```

Key dependencies: PyTorch, NumPy, Numba, SciPy, gurobipy (optional, for exact baselines).

To make the package importable, either:
- Set `PYTHONPATH=.` when running scripts (recommended), or
- Install in dev mode: `pip install -e .`
