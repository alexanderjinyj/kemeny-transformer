#!/usr/bin/env python3
"""
Refactored training script for Kemeny Transformer with cleaner structure.
The train function is decomposed into smaller, manageable helper functions.
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import os
import json
from scipy import stats
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
import random

from kemeny_transformer.model import kemeny_transformer, EmbeddingType
from kemeny_transformer.data.synthesis import DataSynthesis as dsy
from kemeny_transformer.data.synthesis import *
from kemeny_transformer.model.tokenization import KemenyTransformerTokenization as ktt

# Import all the helper functions from the original file
# (In practice, these would be in separate modules)
exec(open('/home/yijun.jin/kemeny-transformer/scripts/train_ddp.py').read())

def train_refactored(args):
    """
    Refactored main training function with cleaner structure.
    """
    # ========== 1. Setup Distributed Environment ==========
    is_ddp, rank, world_size, device = setup_distributed()

    # ========== 2. Set Random Seeds ==========
    random_seed = getattr(args, 'random_seed', 1234)
    actual_seed = set_random_seeds(random_seed, rank)

    if rank == 0:
        print(f"--- Training Setup ---")
        print(f"DDP Enabled: {is_ddp}")
        print(f"World Size:  {world_size}")
        print(f"Device:      {device}")
        print(f"Random Seed: {random_seed} (Rank {rank} actual seed: {actual_seed})")
        print(f"----------------------")

        try:
            print("Loaded configuration:", json.dumps(vars(args), indent=2))
        except Exception as e:
            print(f"Could not print args: {e}")

    # ========== 3. Parse Configuration ==========
    parsed_embedding_type = parse_embedding_type(args, rank)

    # ========== 4. Initialize Models ==========
    model_train, model_baseline = setup_models(args, parsed_embedding_type, device)

    # ========== 5. Setup Data Synthesis ==========
    data_synthesis = dsy(random_seed=actual_seed)

    # ========== 6. Setup Tokenizer ==========
    tokenizer = setup_tokenizer(args, rank)

    # ========== 7. Setup Optimizer and Scheduler ==========
    optimizer = torch.optim.Adam(model_train.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.lr * 0.1)

    # ========== 8. Setup Logging ==========
    time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
    validation_log_handle, rollout_log_handle = setup_logging(args, rank, time_stamp)

    # ========== 9. Load Validation Data ==========
    validation_data = None
    if rank == 0:
        if hasattr(args, 'validation_data_dirs') or hasattr(args, 'validation_data_dir'):
            if hasattr(args, 'validation_data_dirs'):
                validation_data = load_validation_data(args, tokenizer, device,
                                                      embedding_type=parsed_embedding_type)
            elif hasattr(args, 'validation_data_dir'):
                args.validation_data_dirs = [args.validation_data_dir]
                validation_data = load_validation_data(args, tokenizer, device,
                                                      embedding_type=parsed_embedding_type)
        else:
            print("Warning: 'validation_data_dirs' not specified in config. Skipping custom validation.")

    # ========== 10. Load Checkpoint ==========
    checkpoint_file = getattr(args, 'checkpoint_file', None)
    if checkpoint_file:
        checkpoint_dir = getattr(args, 'checkpoint_dir', 'checkpoints')
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)

    epoch_ckpt, tot_time_ckpt, training_phase, running_mean_ckpt, running_std_ckpt = \
        load_checkpoint(checkpoint_file, device, model_train, model_baseline, optimizer, args)

    # ========== 11. Wrap Models with DDP if Needed ==========
    if is_ddp:
        model_train = DDP(model_train, device_ids=[rank], find_unused_parameters=True)
        model_baseline = DDP(model_baseline, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        print(f"Model initialized with {sum(p.numel() for p in model_train.parameters())} parameters.")

    # ========== 12. Setup Training Parameters ==========
    grad_acc_steps = getattr(args, 'gradient_accumulation_steps', 1)
    if rank == 0:
        print(f"Gradient Accumulation Steps: {grad_acc_steps}")

    # Advantage normalization setup
    advantage_norm_type = getattr(args, 'advantage_normalization', 'none').lower()
    running_stats_momentum = getattr(args, 'running_std_momentum', 0.99)
    running_mean = running_mean_ckpt if running_mean_ckpt is not None else 0.0
    running_std = running_std_ckpt if running_std_ckpt is not None else 1.0

    if rank == 0:
        print(f"Advantage Normalization: {advantage_norm_type}")
        print(f"Running Stats - Mean: {running_mean:.4f}, Std: {running_std:.4f}")

    # ========== 13. Training Loop ==========
    start_training_time_sec = time.time()

    for epoch in range(epoch_ckpt, args.nb_epochs):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting Epoch: {epoch}")
            print(f"{'='*60}")

        epoch_start_time_sec = time.time()
        model_train.train()
        optimizer.zero_grad()

        epoch_losses = []
        epoch_advantages = []

        # ========== Training Steps ==========
        for step in range(1, args.nb_batch_per_epoch + 1):
            # Generate batch
            batch_rankings = generate_training_batch(args, data_synthesis, parsed_embedding_type)

            # Perform training step
            kemeny_dist_train, kemeny_dist_baseline, sum_log_prob = \
                perform_training_step(model_train, model_baseline, batch_rankings,
                                    tokenizer, parsed_embedding_type, device, training_phase)

            # Compute advantage
            advantage, running_mean, running_std = \
                compute_advantage(kemeny_dist_train, kemeny_dist_baseline, advantage_norm_type,
                                running_mean, running_std, running_stats_momentum, is_ddp)

            # Compute loss (REINFORCE)
            loss = torch.mean(advantage.detach() * sum_log_prob)
            loss = loss / grad_acc_steps
            loss.backward()

            # Gradient accumulation
            if step % grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Logging
            epoch_losses.append(loss.item() * grad_acc_steps)
            epoch_advantages.append(torch.mean(advantage).item())

            if rank == 0 and step % 50 == 0:
                current_total_time = (time.time() - start_training_time_sec) + tot_time_ckpt
                print(f'Epoch:{epoch}/{args.nb_epochs} | Batch: {step}/{args.nb_batch_per_epoch} | '
                      f'Avg Advantage: {np.mean(epoch_advantages):.4f} | '
                      f'Loss: {loss.item() * grad_acc_steps:.4f} | '
                      f'Time: {current_total_time/60:.2f}min')

        # ========== End of Epoch ==========
        avg_epoch_loss = np.mean(epoch_losses)
        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Training Phase: {training_phase}")

        # ========== Validation ==========
        if rank == 0 and validation_data is not None:
            print("\nRunning validation...")
            model_to_validate = model_train.module if is_ddp else model_train
            run_validation(
                model=model_to_validate,
                validation_data=validation_data,
                dsy=data_synthesis,
                device=device,
                epoch=epoch,
                epoch_start_time=epoch_start_time_sec,
                total_start_time_sec=start_training_time_sec,
                tot_time_ckpt=tot_time_ckpt,
                validation_log_handle=validation_log_handle
            )

        # ========== Baseline Evaluation and Update ==========
        print("\nEvaluating baseline...")
        update_baseline, training_phase = \
            evaluate_baseline_update(args, model_train, model_baseline, data_synthesis,
                                   tokenizer, parsed_embedding_type, device, world_size,
                                   is_ddp, rank, training_phase)

        if update_baseline:
            if rank == 0:
                print("  > Updating baseline model with current weights")

            # Get state dict from training model
            train_state = model_train.module.state_dict() if is_ddp else model_train.state_dict()

            # Update baseline
            if is_ddp:
                model_baseline.module.load_state_dict(train_state)
            else:
                model_baseline.load_state_dict(train_state)

        # Log rollout results
        if rank == 0 and rollout_log_handle:
            time_one_epoch = time.time() - epoch_start_time_sec
            time_tot = (time.time() - start_training_time_sec) + tot_time_ckpt
            rollout_log_record = (
                f'Epoch: {epoch}, epoch time: {time_one_epoch/60:.2f}min, '
                f'tot time: {time_tot/86400:.2f}day, '
                f'phase: {training_phase}, baseline_updated: {update_baseline}'
            )
            rollout_log_handle.write(rollout_log_record + '\n')
            rollout_log_handle.flush()

        # ========== Checkpointing ==========
        save_every = getattr(args, 'save_every', 5)
        checkpoint_dir = getattr(args, 'checkpoint_dir', 'checkpoints')

        if rank == 0 and epoch % save_every == 0 and epoch > 0:
            current_total_time = (time.time() - start_training_time_sec) + tot_time_ckpt

            os.makedirs(checkpoint_dir, exist_ok=True)
            save_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl')

            save_checkpoint(save_path, epoch, model_train, model_baseline, optimizer,
                          training_phase, current_total_time, random_seed,
                          running_mean, running_std, is_ddp)

        # ========== Learning Rate Scheduling ==========
        scheduler.step()

        # ========== Synchronize Processes ==========
        if is_ddp:
            dist.barrier()

    # ========== Training Complete ==========
    if rank == 0:
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)

        if validation_log_handle:
            validation_log_handle.close()
        if rollout_log_handle:
            rollout_log_handle.close()

    if is_ddp:
        ddp_cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kemeny Transformer Training (Refactored)")
    parser.add_argument('--config_file', type=str, required=True,
                       help='Path to the JSON configuration file.')
    cli_args = parser.parse_args()

    # Load configuration from JSON file
    try:
        with open(cli_args.config_file, 'r') as f:
            config_args_dict = json.load(f)

        # Convert config dict to an object
        class ConfigObject:
            def __init__(self, **entries):
                self.__dict__.update(entries)

            def __getattr__(self, name):
                return self.__dict__.get(name, None)

        args_obj = ConfigObject(**config_args_dict)
        train_refactored(args_obj)

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {cli_args.config_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {cli_args.config_file}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if dist.is_initialized():
            ddp_cleanup()
        exit(1)