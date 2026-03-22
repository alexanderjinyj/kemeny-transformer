import pickle
import json
import torch
import torch.nn as nn
from time import time
import numpy as np
import os
from pathlib import Path
from kemeny_transformer.model import kemeny_transformer, EmbeddingType, DotDict
from kemeny_transformer.model.tokenization import KemenyTransformerTokenization


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_test_datasets(test_dir):
    """Load all .npy files from a test directory"""
    test_dir = Path(test_dir)
    datasets = {}
    for npy_file in sorted(test_dir.glob("**/*.npy")):
        rel_path = npy_file.relative_to(test_dir)
        dataset = np.load(str(npy_file))
        datasets[str(rel_path)] = dataset
        print(f"  Loaded {rel_path}: shape {dataset.shape}")
    return datasets


def test_on_dataset(model, tokenizer, dataset, args, device):
    """Test model on a single dataset using greedy decoding"""
    # dataset shape: (num_samples, num_voters, num_candidates)
    dataset_size = dataset.shape[0]
    num_voters = dataset.shape[1]
    num_candidates = dataset.shape[2]

    print(f"  Dataset: {dataset_size} samples, {num_voters} voters, {num_candidates} candidates")

    # Store results
    all_rankings = []
    all_log_probs = []

    model.eval()
    with torch.no_grad():
        start_time = time()

        # Process in batches
        for batch_start in range(0, dataset_size, args.bsz):
            batch_end = min(batch_start + args.bsz, dataset_size)
            batch_data = dataset[batch_start:batch_end]

            print(f"    Processing batch {batch_start//args.bsz + 1}/{(dataset_size + args.bsz - 1)//args.bsz}")

            # Convert batch to list of arrays
            batch_list = [batch_data[i] for i in range(len(batch_data))]

            # Tokenize the batch
            padded_batch, padding_mask, voter_mask = tokenizer.tokenize(
                batch_list,
                embedding_type=args.embedding_type
            )

            # Move to device
            padded_batch = padded_batch.to(device)
            padding_mask = padding_mask.to(device)
            voter_mask = voter_mask.to(device) if voter_mask is not None else None

            # Forward pass (greedy/deterministic)
            final_rankings, avg_log_probs, output_mask = model(
                padded_batch,
                padding_mask,
                voter_mask=voter_mask,
                deterministic=True
            )

            # Store results
            all_rankings.append(final_rankings.cpu())
            all_log_probs.append(avg_log_probs.cpu())

        elapsed_time = time() - start_time

    # Concatenate all batches
    all_rankings = torch.cat(all_rankings, dim=0)
    all_log_probs = torch.cat(all_log_probs, dim=0)

    print(f"  Completed in {elapsed_time:.2f}s ({elapsed_time/dataset_size:.4f}s per sample)")

    return {
        'final_rankings': all_rankings.numpy(),
        'avg_log_probs': all_log_probs.numpy(),
        'inference_time': elapsed_time,
        'time_per_sample': elapsed_time / dataset_size,
        'num_samples': dataset_size,
        'num_voters': num_voters,
        'num_candidates': num_candidates,
    }


def main():
    # Configuration
    config_file = "/home/yijun.jin/kemenyTransformer/args_700k_linear_various_voter_various_items_mix_batch.conf"
    checkpoint_file = "/home/yijun.jin/kemenyTransformer/kemeny_transformer_checkpoint/various_input_kemeny_transformer/linear_embedding/700k_various_voters_various_items_mix_batch_3_encoder_2_decoder_kemeny_transformer_without_guide/checkpoint_epoch_650.pkl"

    test_dirs = [
        "/home/yijun.jin/kemenyTransformer/test_dataset/test_dataset_jiggling",
        "/home/yijun.jin/kemenyTransformer/test_dataset/test_dataset_random",
        "/home/yijun.jin/kemenyTransformer/test_dataset/test_dataset_repeat",
    ]

    # Load configuration
    print("="*80)
    print("Loading configuration from:", config_file)
    print("="*80)
    config = load_config(config_file)

    # Create args object
    args = DotDict()
    args.dim_input = config['dim_input']
    args.dim_emb = config['dim_emb']
    args.dim_ff = config['dim_ff']
    args.numb_heads = config['numb_heads']
    args.numb_layers_decoder = config['numb_layers_decoder']
    args.numb_layers_encoder = config['numb_layers_encoder']
    args.max_len_PE = config['max_len_PE']
    args.batchnorm = config['batchnorm']
    args.gpu_id = config['gpu_id']
    args.conv_out_channels = config.get('conv_out_channels', 64)
    args.normalize_input = config.get('normalize_input', 'False') == 'True'

    # Get embedding type
    embedding_type_str = config.get('embedding_type', 'linear')
    if embedding_type_str.lower() == 'linear':
        args.embedding_type = EmbeddingType.LINEAR
    elif embedding_type_str.lower() == 'conv':
        args.embedding_type = EmbeddingType.CONV
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type_str}")

    # Testing parameters
    args.bsz = 128  # Batch size for testing

    print("\nModel configuration:")
    print(f"  Embedding type: {embedding_type_str}")
    print(f"  dim_input: {args.dim_input}")
    print(f"  dim_emb: {args.dim_emb}")
    print(f"  dim_ff: {args.dim_ff}")
    print(f"  numb_heads: {args.numb_heads}")
    print(f"  numb_layers_encoder: {args.numb_layers_encoder}")
    print(f"  numb_layers_decoder: {args.numb_layers_decoder}")
    print(f"  batchnorm: {args.batchnorm}")
    print(f"  normalize_input: {args.normalize_input}")
    print(f"  gpu_id: {args.gpu_id}")
    print(f"  batch_size: {args.bsz}")

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f'\nGPU: {torch.cuda.get_device_name(0)} (ID: {args.gpu_id})')
        print(f'Number of GPUs: {torch.cuda.device_count()}')
    else:
        device = torch.device("cpu")
        print('\nUsing CPU')

    # Initialize model
    print(f"\n{'='*80}")
    print("Initializing model...")
    print(f"{'='*80}")

    model = kemeny_transformer(
        embedding_type=args.embedding_type,
        input_dim=args.dim_input,
        embedding_dim=args.dim_emb,
        dim_ff=args.dim_ff,
        numb_heads=args.numb_heads,
        numb_layers_decoder=args.numb_layers_decoder,
        numb_layers_encoder=args.numb_layers_encoder,
        max_len_PE=args.max_len_PE,
        conv_out_channels=args.conv_out_channels,
        batchnorm=args.batchnorm
    )

    model = model.to(device)

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)

    # Load model state dict
    if 'model_baseline_state_dict' in checkpoint:
        print("  Loading from 'model_baseline_state_dict'")
        model.load_state_dict(checkpoint['model_baseline_state_dict'])
        print("  Loaded from 'model_baseline_state_dict'")
    elif 'model_train_state_dict' in checkpoint:
        print("  Loading from 'model_train_state_dict'")
        model.load_state_dict(checkpoint['model_train_state_dict'])
        print("  Loaded from 'model_train_state_dict'")
    else:
        raise ValueError(f"Could not find model state dict. Keys: {checkpoint.keys()}")

    epoch_ckpt = checkpoint.get('epoch', -1) + 1
    tot_time_ckpt = checkpoint.get('tot_time', 0)
    print(f"  Checkpoint epoch: {epoch_ckpt}")
    print(f"  Training time: {tot_time_ckpt/3600/24:.3f} days")

    del checkpoint

    # Setup multi-GPU if available (after loading checkpoint)
    if torch.cuda.device_count() > 1:
        print(f"\nUsing DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Initialize tokenizer
    print(f"\nInitializing tokenizer (max_voters={args.dim_input}, normalize={args.normalize_input})")
    tokenizer = KemenyTransformerTokenization(
        max_voters=args.dim_input,
        pad_value=0.0,
        normalize_input=args.normalize_input
    )

    # Create output directory
    output_dir = Path("test_results/700k_various_voters_various_items_epoch_500")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # Test on each directory
    for test_dir in test_dirs:
        dir_name = Path(test_dir).name
        print(f"\n{'='*80}")
        print(f"Testing on: {dir_name}")
        print(f"{'='*80}")

        # Load all datasets from this directory
        print("\nLoading datasets:")
        datasets = load_test_datasets(test_dir)

        # Create subdirectory for this test type
        test_output_dir = output_dir / dir_name
        test_output_dir.mkdir(parents=True, exist_ok=True)

        # Test on each dataset
        for dataset_name, dataset in datasets.items():
            print(f"\n{'-'*80}")
            print(f"Testing: {dataset_name}")
            print(f"{'-'*80}")

            try:
                results = test_on_dataset(model, tokenizer, dataset, args, device)

                # Save results
                output_file = test_output_dir / f"result_{Path(dataset_name).stem}.pkl"
                with open(output_file, "wb") as fp:
                    pickle.dump(results, fp, protocol=4)
                print(f"  ✓ Saved to: {output_file}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n{'='*80}")
    print("Testing completed!")
    print(f"All results saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
