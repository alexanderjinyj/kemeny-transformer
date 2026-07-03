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
#from DetKiwiSort import BatchDetKwikSort as bdks
from kemeny_transformer.data.synthesis import DataSynthesis as dsy
from kemeny_transformer.data.synthesis import *
from kemeny_transformer.model.tokenization import KemenyTransformerTokenization as ktt

# --- Seed Setting Function ---
def set_random_seeds(seed, rank=0):
    """
    Sets random seeds for reproducibility across all libraries.
    Each rank gets a different seed based on the base seed + rank.
    """
    actual_seed = seed + rank
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    torch.cuda.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)
    # For CUDNN backend
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return actual_seed

# --- DDP Helper Functions ---
def ddp_setup():
    """Initializes the distributed process group."""
    # Ensure environment variables 'MASTER_ADDR' and 'MASTER_PORT' are set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="nccl")
    # LOCAL_RANK is set by the torchrun/torch.distributed.launch utility
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()


def clean_padded_permutations(permutation_tensor):
    """
    Converts a padded permutation tensor to a list of numpy arrays,
    with padding (assumed to be 0) removed.
    """
    batch_list_np = []
    # Move tensor to CPU and convert to numpy for iteration
    # Use .detach() to avoid issues with gradient tracking
    permutation_np_batch = permutation_tensor.cpu().detach().numpy()
    for i in range(permutation_np_batch.shape[0]):
        # Get the current row
        row = permutation_np_batch[i]
        # Remove all 0 values
        cleaned_row = row[row != 0]
        batch_list_np.append(cleaned_row)
    return batch_list_np

# --- Helper Function: Load Validation Data ---

def load_single_dataset_type(data_dir, dataset_type, tokenizer, device, embedding_type):
    """
    Helper function to load a single dataset type (random, repeat, or jiggling).
    Returns the processed data dictionary or None if the dataset doesn't exist.
    """
    data_file_path = os.path.join(data_dir, f'validate_dataset_{dataset_type}.npy')
    if not os.path.exists(data_file_path):
        return None

    # Load raw data
    val_raw = np.load(data_file_path, allow_pickle=True)

    # Load optimal distances if available
    dist_file_path = os.path.join(data_dir, f'validate_dataset_{dataset_type}_kemeny_optimal_ranking_distance_gurobi.npy')
    if os.path.exists(dist_file_path):
        val_dist = np.load(dist_file_path)
    else:
        val_dist = np.zeros(val_raw.shape[0])

    # Convert to list format
    list_val = [val_raw[i] for i in range(val_raw.shape[0])]

    # Tokenize
    val_token, val_mask, val_voter_mask = tokenizer.tokenize(
        batch_base_rankings=list_val, embedding_type=embedding_type)

    # Move to device
    val_token = val_token.to(device)
    val_mask = val_mask.to(device)
    val_voter_mask = val_voter_mask.to(device)

    return {
        "raw": list_val,
        "token": val_token,
        "mask": val_mask,
        "voter_mask": val_voter_mask,
        "optimal_dist": torch.from_numpy(val_dist).to(device, dtype=torch.float32)
    }


def load_validation_data(args, tokenizer, device, embedding_type="linear"):
    """
    Loads the validation datasets from multiple directories.
    Each directory should be in the format n_voters_m_items.
    This should only be called on rank 0.
    """
    print("Loading validation data...")

    # Check if validation_data_dirs is provided and is a list
    if not hasattr(args, 'validation_data_dirs') or args.validation_data_dirs is None:
        # Check for single dir for backwards compatibility
        if hasattr(args, 'validation_data_dir') and args.validation_data_dir is not None:
            validation_data_dirs = [args.validation_data_dir]
        else:
            print("Warning: 'validation_data_dirs' not specified in config. Skipping custom validation.")
            return None
    else:
        validation_data_dirs = args.validation_data_dirs
        if not isinstance(validation_data_dirs, list):
            # If it's a single string, convert to list
            validation_data_dirs = [validation_data_dirs]

    all_validation_data = {}

    for data_dir in validation_data_dirs:
        try:
            print(f"Loading validation data from {data_dir}")

            # Extract n_voters and m_items from directory name
            dir_name = os.path.basename(data_dir)
            # Expected format: n_voters_m_items
            parts = dir_name.split('_')
            if len(parts) >= 4 and parts[1] == 'voters' and parts[3] == 'items':
                n_voters = parts[0]
                m_items = parts[2]
                dir_key = f"{n_voters}_voters_{m_items}_items"
            else:
                dir_key = dir_name
                print(f"Warning: Directory name {dir_name} doesn't match expected format n_voters_m_items")

            # Load data for each type (random, repeat, jiggling)
            dir_data = {}

            # Define dataset types to load
            dataset_types = ['random', 'repeat', 'jiggling']

            for dataset_type in dataset_types:
                dataset = load_single_dataset_type(data_dir, dataset_type, tokenizer, device, embedding_type)
                if dataset is not None:
                    dir_data[dataset_type] = dataset
                    print(f"  Loaded {dataset_type} dataset for {dir_key}")
                else:
                    print(f"  No {dataset_type} dataset found for {dir_key}")

            if dir_data:
                all_validation_data[dir_key] = dir_data
                print(f"Validation data loaded successfully for {dir_key}")
            else:
                print(f"Warning: No validation data found in {data_dir}")

        except FileNotFoundError as e:
            print(f"Error: Validation file not found in {data_dir}. {e}")
            continue
        except Exception as e:
            print(f"Error loading validation data from {data_dir}: {e}")
            continue

    if not all_validation_data:
        print("No validation data loaded from any directory.")
        return None

    # For backwards compatibility, if only one directory, return old format
    if len(all_validation_data) == 1:
        # Return the data for the single directory in the old format
        single_dir_data = list(all_validation_data.values())[0]
        return single_dir_data

    return all_validation_data

# --- Helper Function: Run Validation ---

def run_validation(model, validation_data, dsy, device, epoch, epoch_start_time, total_start_time_sec, tot_time_ckpt, validation_log_handle):
    """
    Runs model evaluation on the validation datasets and logs results.
    Now handles multiple validation directories.
    This should only be called on rank 0.
    """
    if validation_data is None:
        print("Validation data is None. Skipping validation.")
        return # Skip if data wasn't loaded

    model.eval()

    def get_metrics(data_dict):
        with torch.no_grad():
            orders_train, _,_ = model(x=data_dict["token"], padding_mask=data_dict["mask"], deterministic=True)

        cleaned_orders = clean_padded_permutations(orders_train)
        rankings_train = order_to_rank_batch(cleaned_orders)

        kemeny_dist_train = torch.from_numpy(kemeny_distance_batch(data_dict["raw"], rankings_train)).to(device, dtype=torch.float32)

        kemeny_gap = torch.sub(kemeny_dist_train, data_dict["optimal_dist"])

        # Calculate gap percentage, replacing inf with 0 (for cases where optimal_dist is 0)
        kemeny_gap_percent = torch.div(kemeny_gap, data_dict["optimal_dist"])
        kemeny_gap_percent = torch.where(torch.isinf(kemeny_gap_percent), 0.0, kemeny_gap_percent)

        return torch.mean(kemeny_gap), torch.nanmean(kemeny_gap_percent), kemeny_dist_train

    time_one_epoch = time.time() - epoch_start_time
    time_tot = (time.time() - total_start_time_sec) + tot_time_ckpt

    # Start building the validation record
    record_of_epoch_validate = (
        f'Epoch: {epoch}, epoch time: {time_one_epoch/60:.2f}min, tot time: {time_tot/86400:.2f}day'
    )

    # Check if validation_data is in the old format (single directory with random/repeat/jiggling keys)
    # or new format (multiple directories)
    if "random" in validation_data and "repeat" in validation_data and "jiggling" in validation_data:
        # Old format - single directory
        try:
            gap_random_mean, gap_perc_random_mean, dist_random = get_metrics(validation_data["random"])
            gap_repeat_mean, gap_perc_repeat_mean, dist_repeat = get_metrics(validation_data["repeat"])
            gap_jiggling_mean, gap_perc_jiggling_mean, dist_jiggling = get_metrics(validation_data["jiggling"])

            record_of_epoch_validate += (
                f', gap_random: {gap_random_mean:.4f},{gap_perc_random_mean:.4f}, '
                f'gap_repeat: {gap_repeat_mean:.4f},{gap_perc_repeat_mean:.4f}, '
                f'gap_jiggling: {gap_jiggling_mean:.4f},{gap_perc_jiggling_mean:.4f}'
            )
        except Exception as e:
            print(f"Error during validation: {e}")
            record_of_epoch_validate += f', error: {str(e)}'
    else:
        # New format - multiple directories
        for dir_key, dir_data in validation_data.items():
            try:
                print(f"Validating on {dir_key}...")

                # Initialize metrics for this directory
                dir_metrics = []

                # Process random data if available
                if "random" in dir_data:
                    gap_random_mean, gap_perc_random_mean, dist_random = get_metrics(dir_data["random"])
                    dir_metrics.append(f"random:{gap_random_mean:.4f}")
                else:
                    dir_metrics.append("random:N/A")

                # Process repeat data if available
                if "repeat" in dir_data:
                    gap_repeat_mean, gap_perc_repeat_mean, dist_repeat = get_metrics(dir_data["repeat"])
                    dir_metrics.append(f"repeat:{gap_repeat_mean:.4f}")
                else:
                    dir_metrics.append("repeat:N/A")

                # Process jiggling data if available
                if "jiggling" in dir_data:
                    gap_jiggling_mean, gap_perc_jiggling_mean, dist_jiggling = get_metrics(dir_data["jiggling"])
                    dir_metrics.append(f"jiggling:{gap_jiggling_mean:.4f}")
                else:
                    dir_metrics.append("jiggling:N/A")

                # Add this directory's results to the record
                record_of_epoch_validate += f', {dir_key}: [{", ".join(dir_metrics)}]'

            except Exception as e:
                print(f"Error during validation for {dir_key}: {e}")
                record_of_epoch_validate += f', {dir_key}: [error]'

    print(record_of_epoch_validate)
    if validation_log_handle:
        validation_log_handle.write(record_of_epoch_validate + '\n')
        validation_log_handle.flush() # Ensure it's written immediately


# --- Training Helper Functions ---

def setup_distributed():
    """
    Setup distributed training environment.
    Returns: is_ddp, rank, world_size, device
    """
    is_ddp = False
    rank = 0
    world_size = 1
    device = None

    # Check if WORLD_SIZE env var is set (indicating torchrun)
    if "WORLD_SIZE" in os.environ:
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size > 1:
                is_ddp = True
                ddp_setup()
                rank = dist.get_rank()
                device = torch.device(f"cuda:{rank}")
            else:
                # Running with torchrun --nproc_per_node=1
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except ValueError:
            print("Warning: WORLD_SIZE env var is not an integer. Defaulting to single-GPU mode.")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        # Standard python execution
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return is_ddp, rank, world_size, device


def parse_embedding_type(args, rank):
    """Parse and validate embedding type from config."""
    parsed_embedding_type = EmbeddingType.LINEAR  # Default
    if args.embedding_type:  # Check if it's not None or empty
        try:
            # Get the Enum member (e.g., EmbeddingType.LINEAR) from the string "LINEAR"
            parsed_embedding_type = getattr(EmbeddingType, args.embedding_type.upper())
            if rank == 0:
                print(f"Using embedding type: {args.embedding_type.upper()}")
        except (AttributeError, TypeError):
            if rank == 0:
                print(f"Warning: Unknown or invalid embedding_type '{args.embedding_type}'. Defaulting to LINEAR.")
    elif rank == 0:
        print("Warning: 'embedding_type' not specified in config. Defaulting to LINEAR.")

    return parsed_embedding_type


def setup_models(args, parsed_embedding_type, device):
    """
    Initialize training and baseline models.
    Returns: model_train, model_baseline
    """
    # Automatically set dim_input to max(num_voters_range) if available
    if hasattr(args, 'num_voters_range') and args.num_voters_range is not None:
        args.dim_input = max(args.num_voters_range)
        print(f"Auto-set dim_input = {args.dim_input} from max(num_voters_range)")

    # Create the training model
    model_train = kemeny_transformer(
        embedding_type=parsed_embedding_type,
        input_dim=args.dim_input,
        embedding_dim=args.dim_emb,
        dim_ff=args.dim_ff,
        numb_heads=args.numb_heads,
        numb_layers_decoder=args.numb_layers_decoder,
        numb_layers_encoder=args.numb_layers_encoder,
        max_len_PE=args.max_len_PE,
        conv_out_channels=args.conv_out_channels,
        batchnorm=args.batchnorm,
    ).to(device)

    # Initialize baseline model (same architecture)
    model_baseline = kemeny_transformer(
        embedding_type=parsed_embedding_type,
        input_dim=args.dim_input,
        embedding_dim=args.dim_emb,
        dim_ff=args.dim_ff,
        numb_heads=args.numb_heads,
        numb_layers_decoder=args.numb_layers_decoder,
        numb_layers_encoder=args.numb_layers_encoder,
        max_len_PE=args.max_len_PE,
        conv_out_channels=args.conv_out_channels,
        batchnorm=args.batchnorm,
    ).to(device)

    return model_train, model_baseline


def setup_tokenizer(args, rank):
    """Setup tokenizer with proper configuration."""
    try:
        if args.num_voters_range is not None and len(args.num_voters_range) == 2:
            tokenizer_max_voters = args.num_voters_range[1]
        elif args.num_voters is not None:
            tokenizer_max_voters = args.num_voters
        else:
            tokenizer_max_voters = 100
    except (TypeError, IndexError, AttributeError):
        if rank == 0:
            print("Warning: 'num_voters_range' not found or invalid in config. Defaulting tokenizer max_voters to 100.")
        tokenizer_max_voters = 100

    print(f"{tokenizer_max_voters=}")

    normalize_input = getattr(args, 'normalize_input', None)
    if normalize_input is not None:
        normalize_input = str(normalize_input).lower() == 'true'
    else:
        normalize_input = False

    if rank == 0:
        print(f"Normalize Input: {normalize_input}")

    tokenizer = ktt(max_voters=tokenizer_max_voters, pad_value=0.0, normalize_input=normalize_input)
    return tokenizer


def setup_logging(args, rank, time_stamp):
    """Setup logging files for validation and rollout."""
    validation_log_handle = None
    rollout_log_handle = None

    if rank == 0:
        log_dir = getattr(args, 'validate_log_dir', 'validate_logs/default_validate_logs')
        os.makedirs(log_dir, exist_ok=True)
        n_str = f"-n{args.nb_candidates}" if hasattr(args, 'nb_candidates') and args.nb_candidates else ""
        validation_log_name = f'validation_{time_stamp}{n_str}.txt'
        validation_log_file_path = os.path.join(log_dir, validation_log_name)
        print(f"Logging validation results to: {validation_log_file_path}")
        validation_log_handle = open(validation_log_file_path, "a", 1)
        validation_log_handle.write(time_stamp + '\n\n')

        rollout_log_dir = getattr(args, 'roll_out_log_dir', 'roll_out_logs/default_rollout_logs')
        os.makedirs(rollout_log_dir, exist_ok=True)
        rollout_validation_log_name = f'roll_out_{time_stamp}{n_str}.txt'
        rollout_validation_log_file_path = os.path.join(rollout_log_dir, rollout_validation_log_name)
        print(f"Logging rollout results to: {rollout_validation_log_file_path}")
        rollout_log_handle = open(rollout_validation_log_file_path, "a", 1)
        rollout_log_handle.write(time_stamp + '\n\n')

        # Write hyperparameters to log files
        try:
            for arg in vars(args):
                hyper_param_val = getattr(args, arg)
                validation_log_handle.write(f"{arg}={hyper_param_val}\n")
                rollout_log_handle.write(f"{arg}={hyper_param_val}\n")
        except Exception as e:
            validation_log_handle.write(f"Could not write args: {e}\n")
            rollout_log_handle.write(f"Could not write args: {e}\n")

        validation_log_handle.write('\n\n')
        validation_log_handle.flush()
        rollout_log_handle.write('\n\n')
        rollout_log_handle.flush()

    return validation_log_handle, rollout_log_handle


def generate_training_batch(args, data_synthesis, embedding_type):
    """Generate a batch of training data based on the specified method."""
    sample_kwargs = {}
    sample_items_from_range = getattr(args, 'sample_items_from_range', None)
    sample_voters_from_range = getattr(args, 'sample_voters_from_range', None)

    if sample_items_from_range is not None:
        sample_kwargs['sample_items_from_range'] = str(sample_items_from_range).lower() == 'true'
    if sample_voters_from_range is not None:
        sample_kwargs['sample_voters_from_range'] = str(sample_voters_from_range).lower() == 'true'

    if args.data_generation_method == "Mallows":
        batch_rankings, _, _ = data_synthesis.batch_generate_base_rankings_Mallows_all_same_shape_vcode(
            bsz=args.bsz,
            num_voters_range=args.num_voters_range,
            num_items_range=args.num_items_range,
            phi_range=args.phi_range,
            **sample_kwargs
        )
    elif args.data_generation_method == "random":
        batch_rankings, _ = data_synthesis.generate_batch_dataset_random_from_range(
            args.bsz,
            args.num_voters_range,
            args.num_items_range,
            **sample_kwargs
        )
    elif args.data_generation_method == "fine_tuning":
        batch_rankings, _ = data_synthesis.generate_batch_instances_fine_tuning(
            args.bsz,
            args.num_voters_range,
            args.num_items_range,
            **sample_kwargs
        )
    elif args.data_generation_method == "fine_tuning_mix":
        batch_rankings, _ = data_synthesis.generate_mix_batch_instances_fine_tuning(
            args.bsz,
            args.num_voters_range,
            args.num_items_range,
            **sample_kwargs
        )
    elif args.data_generation_method == "random_mix":
        batch_rankings, _ = data_synthesis.generate_mix_batch_dataset_random_from_range(
            args.bsz,
            args.num_voters_range,
            args.num_items_range,
            **sample_kwargs
        )
    else:
        raise ValueError(f"Unknown data_generation_method: {args.data_generation_method}")

    if isinstance(batch_rankings, np.ndarray):
        batch_rankings = [batch_rankings[i] for i in range(batch_rankings.shape[0])]

    return batch_rankings


def compute_advantage(kemeny_dist_train, kemeny_dist_baseline, advantage_norm_type,
                     running_mean, running_std, running_stats_momentum, is_ddp):
    """Compute and normalize advantage for REINFORCE."""
    advantage = kemeny_dist_train - kemeny_dist_baseline

    # Advantage normalization (configurable)
    if advantage.shape[0] > 1:
        if advantage_norm_type == "running":
            # Running mean and std: full standardization with EMA
            batch_mean_tensor = advantage.mean()
            batch_std_tensor = advantage.std()

            # DDP Sync: average stats across all GPUs
            if is_ddp:
                dist.all_reduce(batch_mean_tensor, op=dist.ReduceOp.AVG)
                dist.all_reduce(batch_std_tensor, op=dist.ReduceOp.AVG)

            batch_mean = batch_mean_tensor.item()
            batch_std = batch_std_tensor.item()

            # Update running statistics with EMA
            running_mean = running_stats_momentum * running_mean + (1 - running_stats_momentum) * batch_mean
            running_std = running_stats_momentum * running_std + (1 - running_stats_momentum) * batch_std

            # Normalize using running stats
            advantage = (advantage - running_mean) / (running_std + 1e-8)
        elif advantage_norm_type == "batch":
            # Per-batch centering
            adv_mean = advantage.mean()
            adv_std = advantage.std()
            advantage = (advantage - adv_mean) / (adv_std + 1e-8)
        elif advantage_norm_type == "none":
            # No normalization - use raw advantage
            pass
        else:
            # Scale-only (default fallback for unknown types)
            adv_std = advantage.std()
            advantage = advantage / (adv_std + 1e-8)

    return advantage, running_mean, running_std


def load_checkpoint(checkpoint_file, device, model_train, model_baseline, optimizer, args):
    """Load checkpoint if it exists."""
    epoch_ckpt = 0
    tot_time_ckpt = 0
    training_phase = 1
    running_mean_ckpt = None
    running_std_ckpt = None

    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        print(f"Loading checkpoint from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        epoch_ckpt = checkpoint['epoch'] + 1
        tot_time_ckpt = checkpoint.get('tot_time', 0)

        # Load model states
        model_train.load_state_dict(checkpoint['model_train_state_dict'])
        training_phase = checkpoint.get('training_phase', 1)

        if training_phase == 2 and checkpoint.get('model_baseline_state_dict') is not None:
            model_baseline.load_state_dict(checkpoint['model_baseline_state_dict'])
        else:
            model_baseline.load_state_dict(checkpoint['model_train_state_dict'])

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state dict. {e}")

        # Load running stats
        running_mean_ckpt = checkpoint.get('running_mean', None)
        running_std_ckpt = checkpoint.get('running_std', None)

        print(f'Restarting from epoch {epoch_ckpt-1}, time={tot_time_ckpt/60:.3f}min')
        del checkpoint

    elif checkpoint_file is not None:
        print(f"Warning: Checkpoint file {checkpoint_file} not found. Starting from scratch.")

    # Initialize baseline with train weights if no checkpoint
    if epoch_ckpt == 0:
        model_baseline.load_state_dict(model_train.state_dict())
        print("Initialized model_baseline with model_train weights.")

    return epoch_ckpt, tot_time_ckpt, training_phase, running_mean_ckpt, running_std_ckpt


def save_checkpoint(save_path, epoch, model_train, model_baseline, optimizer,
                   training_phase, tot_time, random_seed, running_mean, running_std, is_ddp):
    """Save training checkpoint."""
    # Get unwrapped state dicts
    train_state = model_train.module.state_dict() if is_ddp else model_train.state_dict()
    base_state = None
    if model_baseline:
        base_state = model_baseline.module.state_dict() if is_ddp else model_baseline.state_dict()

    torch.save({
        'epoch': epoch,
        'model_train_state_dict': train_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'training_phase': training_phase,
        'model_baseline_state_dict': base_state,
        'tot_time': tot_time,
        'random_seed': random_seed,
        'running_mean': running_mean,
        'running_std': running_std
    }, save_path)
    print(f"  > Checkpoint saved to {save_path}.")


def perform_training_step(model_train, model_baseline, batch_rankings, tokenizer,
                          parsed_embedding_type, device, training_phase):
    """Perform a single training step."""
    # Tokenize batch
    batch_rankings_token, padding_mask, voter_mask = tokenizer.tokenize(
        batch_rankings, embedding_type=parsed_embedding_type)
    batch_rankings_token = batch_rankings_token.to(device)
    padding_mask = padding_mask.to(device)
    voter_mask = voter_mask.to(device)

    # Policy (Actor) Forward Pass
    orders_train, sum_log_prob, _ = model_train(
        x=batch_rankings_token, padding_mask=padding_mask,
        voter_mask=voter_mask, deterministic=False)

    # Process outputs
    cleaned_orders_train = clean_padded_permutations(orders_train)
    rankings_train = order_to_rank_batch(cleaned_orders_train)

    # Calculate Kemeny distance for actor
    kemeny_dist_train = torch.from_numpy(
        kemeny_distance_batch(batch_rankings, rankings_train)
    ).to(device, dtype=torch.float32)

    # Calculate baseline
    if training_phase == 1:
        # Phase 1: Could use heuristic baseline here
        kemeny_dist_baseline = torch.zeros_like(kemeny_dist_train)
    else:
        # Phase 2: Use self-improving model_baseline
        with torch.no_grad():
            orders_baseline, _, _ = model_baseline(
                x=batch_rankings_token, padding_mask=padding_mask,
                voter_mask=voter_mask, deterministic=True)
            cleaned_orders_baseline = clean_padded_permutations(orders_baseline)
            rankings_baseline = order_to_rank_batch(cleaned_orders_baseline)
            kemeny_dist_baseline = torch.from_numpy(
                kemeny_distance_batch(batch_rankings, rankings_baseline)
            ).to(device, dtype=torch.float32)

    return kemeny_dist_train, kemeny_dist_baseline, sum_log_prob


def evaluate_baseline_update(args, model_train, model_baseline, data_synthesis, tokenizer,
                            parsed_embedding_type, device, world_size, is_ddp, rank,
                            training_phase):
    """Evaluate if baseline should be updated based on rollout performance."""
    model_train.eval()
    if model_baseline:
        model_baseline.eval()

    dist_train_eval_local = np.array([])
    dist_base_eval_local = np.array([])

    with torch.no_grad():
        eval_bsz_per_gpu = args.roll_out_bsz // world_size
        if eval_bsz_per_gpu == 0:
            print(f"Warning: eval_bsz_per_gpu is 0. Skipping baseline evaluation.")
            return False, training_phase

        # Generate evaluation batch
        eval_raw_rankings = generate_training_batch(args, data_synthesis, parsed_embedding_type)

        if eval_raw_rankings is None:
            return False, training_phase

        # Tokenize
        eval_rankings_token, eval_padding_mask, eval_voter_mask = tokenizer.tokenize(
            eval_raw_rankings, embedding_type=parsed_embedding_type)
        eval_rankings_token = eval_rankings_token.to(device)
        eval_padding_mask = eval_padding_mask.to(device)
        eval_voter_mask = eval_voter_mask.to(device)

        # Evaluate train model
        perm_train_eval, _, _ = model_train(
            x=eval_rankings_token, padding_mask=eval_padding_mask,
            voter_mask=eval_voter_mask, deterministic=True)
        cleaned_perm_train = clean_padded_permutations(perm_train_eval)
        rank_train_eval = order_to_rank_batch(cleaned_perm_train)
        dist_train_eval_local = kemeny_distance_batch(eval_raw_rankings, rank_train_eval)

        # Evaluate baseline
        if training_phase == 1:
            # Phase 1: Could use heuristic baseline
            dist_base_eval_local = np.ones_like(dist_train_eval_local) * 1000  # High dummy value
        else:
            perm_base_eval, _, _ = model_baseline(
                x=eval_rankings_token, padding_mask=eval_padding_mask,
                voter_mask=eval_voter_mask, deterministic=True)
            cleaned_perm_base = clean_padded_permutations(perm_base_eval)
            rank_base_eval = order_to_rank_batch(cleaned_perm_base)
            dist_base_eval_local = kemeny_distance_batch(eval_raw_rankings, rank_base_eval)

        # Gather results from all GPUs if DDP
        if is_ddp:
            dist_train_eval_gathered = [np.zeros_like(dist_train_eval_local) for _ in range(world_size)]
            dist_base_eval_gathered = [np.zeros_like(dist_base_eval_local) for _ in range(world_size)]
            dist.all_gather_object(dist_train_eval_gathered, dist_train_eval_local)
            dist.all_gather_object(dist_base_eval_gathered, dist_base_eval_local)
        else:
            dist_train_eval_gathered = [dist_train_eval_local]
            dist_base_eval_gathered = [dist_base_eval_local]

        update_baseline = False
        new_training_phase = training_phase

        if rank == 0:
            dist_train_eval = np.concatenate(dist_train_eval_gathered)
            dist_base_eval = np.concatenate(dist_base_eval_gathered)

            mean_dist_train = np.mean(dist_train_eval)
            mean_dist_base = np.mean(dist_base_eval)

            print(f"  Baseline Eval: Train={mean_dist_train:.4f}, Baseline={mean_dist_base:.4f}")

            if mean_dist_train < mean_dist_base:
                try:
                    t_stat, p_value = stats.ttest_rel(dist_train_eval, dist_base_eval)
                    if t_stat < 0 and p_value / 2 < args.baseline_alpha:
                        print(f"  > Significant improvement (p={p_value/2:.5f})")
                        update_baseline = True
                        if training_phase == 1:
                            new_training_phase = 2
                            print("PHASE 1 COMPLETE: Switching to Phase 2")
                except ValueError as e:
                    print(f"  > T-test failed: {e}")

        # Broadcast update decision to all ranks
        if is_ddp:
            update_tensor = torch.tensor([1 if update_baseline else 0, new_training_phase],
                                       device=device, dtype=torch.int)
            dist.broadcast(update_tensor, src=0)
            update_baseline = bool(update_tensor[0].item())
            new_training_phase = update_tensor[1].item()

    return update_baseline, new_training_phase


# --- Main Training Script ---

def train(args):
    """
    Main training function - refactored for better modularity and readability.

    This function orchestrates the entire training process using helper functions
    for each major component of the training pipeline.
    """
    # ========== 1. Setup Distributed Environment ==========
    is_ddp, rank, world_size, device = setup_distributed()
        try:
            world_size = int(os.environ["WORLD_SIZE"])
            if world_size > 1:
                is_ddp = True
                ddp_setup()
                rank = dist.get_rank()
                device = torch.device(f"cuda:{rank}")
            else:
                # Running with torchrun --nproc_per_node=1
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except ValueError:
            print("Warning: WORLD_SIZE env var is not an integer. Defaulting to single-GPU mode.")
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        # Standard python execution
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- Set Random Seeds ---
    random_seed = getattr(args, 'random_seed', 1234)  # Default seed if not specified
    actual_seed = set_random_seeds(random_seed, rank)

    if rank == 0:
        print(f"--- Training Setup ---")
        print(f"DDP Enabled: {is_ddp}")
        print(f"World Size:  {world_size}")
        print(f"Device:      {device}")
        print(f"Random Seed: {random_seed} (Rank {rank} actual seed: {actual_seed})")
        print(f"----------------------")

    if rank == 0:
        try:
            # vars() works on argparse.Namespace and our custom ConfigObject
            print("Loaded configuration:", json.dumps(vars(args), indent=2))
        except Exception as e:
            print(f"Could not print args: {e}")

    parsed_embedding_type = EmbeddingType.LINEAR # Default
    if args.embedding_type: # Check if it's not None or empty
        try:
            # Get the Enum member (e.g., EmbeddingType.LINEAR) from the string "LINEAR"
            parsed_embedding_type = getattr(EmbeddingType, args.embedding_type.upper())
            if rank == 0:
                print(f"Using embedding type: {args.embedding_type.upper()}")
        except (AttributeError, TypeError):
            if rank == 0:
                print(f"Warning: Unknown or invalid embedding_type '{args.embedding_type}'. Defaulting to LINEAR.")
    elif rank == 0:
        print("Warning: 'embedding_type' not specified in config. Defaulting to LINEAR.")

    # 1. --- Model Initialization ---
    # Automatically set dim_input to max(num_voters_range) if available
    if hasattr(args, 'num_voters_range') and args.num_voters_range is not None:
        args.dim_input = max(args.num_voters_range)
        if rank == 0:
            print(f"Auto-set dim_input = {args.dim_input} from max(num_voters_range)")

    # Always create the base model on the correct device first
    model_train = kemeny_transformer(
        embedding_type=parsed_embedding_type,
        input_dim=args.dim_input,
        embedding_dim=args.dim_emb,
        dim_ff=args.dim_ff,
        numb_heads=args.numb_heads,
        numb_layers_decoder=args.numb_layers_decoder,
        numb_layers_encoder=args.numb_layers_encoder,
        max_len_PE=args.max_len_PE,
        conv_out_channels = args.conv_out_channels,
        batchnorm= args.batchnorm,
    ).to(device)

    # Seed data synthesis per rank using the configured random seed
    data_synthesis=dsy(random_seed=actual_seed)

    try:
        if args.num_voters_range is not None and len(args.num_voters_range) == 2:
            tokenizer_max_voters = args.num_voters_range[1]
        if args.num_voters is not None:
            tokenizer_max_voters = args.num_voters
    except (TypeError, IndexError, AttributeError):
        if rank == 0:
            print("Warning: 'num_voters_range' not found or invalid in config. Defaulting tokenizer max_voters to 100.")
        tokenizer_max_voters = 100
    print(f"{tokenizer_max_voters=}")

    normalize_input = getattr(args, 'normalize_input', None)
    if normalize_input is not None:
         normalize_input = str(normalize_input).lower() == 'true'
    else:
        normalize_input = False
    if rank == 0:
        print(f"Normalize Input: {normalize_input}")

    tokenizer=ktt(max_voters=tokenizer_max_voters, pad_value=0.0, normalize_input=normalize_input)

    # Initialize optimizer with the *base* model parameters
    optimizer = torch.optim.Adam(model_train.parameters(), lr=args.lr)

    # Initialize LR Scheduler (Cosine Annealing)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.nb_epochs, eta_min=args.lr * 0.1)

    # --- Logging, Validation, Checkpoint Vars (Init for all) ---
    model_baseline = kemeny_transformer(
        embedding_type=parsed_embedding_type,
        input_dim=args.dim_input,
        embedding_dim=args.dim_emb,
        dim_ff=args.dim_ff,
        numb_heads=args.numb_heads,
        numb_layers_decoder=args.numb_layers_decoder,
        numb_layers_encoder=args.numb_layers_encoder,
        max_len_PE=args.max_len_PE,
        conv_out_channels = args.conv_out_channels,
        batchnorm= args.batchnorm,
    ).to(device)
    training_phase = 2
    epoch_ckpt = 0
    tot_time_ckpt = 0
    checkpoint_dir = getattr(args, 'checkpoint_dir', 'checkpoints')
    checkpoint_file = getattr(args, 'checkpoint_file', None)
    if checkpoint_file is not None:
        checkpoint_file= os.path.join(checkpoint_dir, checkpoint_file)
    validation_log_handle = None
    rollout_log_handle = None
    validation_data = None

    # --- Setup Logging and Load Validation Data (Rank 0 Only) ---
    if rank == 0:
        time_stamp = datetime.datetime.now().strftime("%y-%m-%d--%H-%M-%S")
        log_dir = getattr(args, 'validate_log_dir', 'validate_logs/default_validate_logs')
        os.makedirs(log_dir, exist_ok=True)
        n_str = f"-n{args.nb_candidates}" if args.nb_candidates else ""
        validation_log_name = f'validation_{time_stamp}{n_str}.txt'
        validation_log_file_path = os.path.join(log_dir, validation_log_name)
        print(f"Logging validation results to: {validation_log_file_path}")
        validation_log_handle = open(validation_log_file_path, "a", 1)
        validation_log_handle.write(time_stamp + '\n\n')

        rollout_log_dir = getattr(args, 'roll_out_log_dir', 'roll_out_logs/default_rollout_logs')
        os.makedirs(rollout_log_dir, exist_ok=True)
        rollout_validation_log_name = f'roll_out_{time_stamp}{n_str}.txt'
        rollout_validation_log_file_path = os.path.join(rollout_log_dir, rollout_validation_log_name)
        print(f"Logging rollout results to: {rollout_validation_log_file_path}")
        rollout_log_handle = open(rollout_validation_log_file_path, "a", 1)
        rollout_log_handle.write(time_stamp + '\n\n')

        try:
            for arg in vars(args):
                hyper_param_val = getattr(args, arg)
                validation_log_handle.write(f"{arg}={hyper_param_val}\n")
                rollout_log_handle.write(f"{arg}={hyper_param_val}\n")
        except Exception as e:
            validation_log_handle.write(f"Could not write args: {e}\n")
            rollout_log_handle.write(f"Could not write args: {e}\n")

        validation_log_handle.write('\n\n')
        validation_log_handle.flush()
        rollout_log_handle.write('\n\n')
        rollout_log_handle.flush()

        if hasattr(args, 'validation_data_dir'):
            validation_data = load_validation_data(args, tokenizer, device, embedding_type=parsed_embedding_type)
        else:
            print("Warning: 'validation_data_dir' not specified in config. Skipping custom validation.")
    # --- End of Rank 0 Setup Block ---

    # --- Checkpoint Loading ---
    # Load state dicts into the *base* models *before* wrapping with DDP
    if checkpoint_file is not None and os.path.exists(checkpoint_file):
        if rank == 0:
            print(f"Loading checkpoint from {checkpoint_file}...")
        checkpoint = torch.load(checkpoint_file, map_location=device)
        epoch_ckpt = checkpoint['epoch'] + 1
        tot_time_ckpt = checkpoint.get('tot_time', 0)

        # Load state dict for the training model
        # The saved state dicts (from module or base model) don't have 'module.' prefix
        model_train.load_state_dict(checkpoint['model_train_state_dict'])

        training_phase = checkpoint['training_phase']

        if training_phase == 2:
            if rank == 0:
                print("Checkpoint is Phase 2, initializing and loading baseline model.")
            model_baseline = kemeny_transformer(
                embedding_type=parsed_embedding_type,
                input_dim=args.dim_input,
                embedding_dim=args.dim_emb,
                dim_ff=args.dim_ff,
                numb_heads=args.numb_heads,
                numb_layers_decoder=args.numb_layers_decoder,
                numb_layers_encoder=args.numb_layers_encoder,
                max_len_PE=args.max_len_PE,
                conv_out_channels = args.conv_out_channels,
                batchnorm= args.batchnorm,
            ).to(device)

            if checkpoint.get('model_baseline_state_dict') is not None:
                model_baseline.load_state_dict(checkpoint['model_baseline_state_dict'])
            else:
                if rank == 0:
                    print("Warning: Checkpoint is Phase 2 but baseline state dict is None. Initializing from model_train.")
                model_baseline.load_state_dict(checkpoint['model_train_state_dict']) # Use train state

        # Load optimizer state *after* models are populated
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Could not load optimizer state dict. {e}. Optimizer will start from scratch.")
        else:
            if rank == 0:
                print("Warning: Optimizer state not found in checkpoint. Optimizer will start from scratch.")

        if rank == 0:
            print(f'Re-start training with saved checkpoint file={checkpoint_file}\n  Checkpoint at epoch={epoch_ckpt-1} and time={tot_time_ckpt/60:.3f}min\n')

        # Load running stats for advantage normalization (with backward compatibility)
        if 'running_mean' in checkpoint:
            running_mean_ckpt = checkpoint['running_mean']
            if rank == 0:
                print(f"Loaded running_mean from checkpoint: {running_mean_ckpt}")
        else:
            running_mean_ckpt = None
            if rank == 0:
                print("Warning: running_mean not found in checkpoint. Will use default initialization.")

        if 'running_std' in checkpoint:
            running_std_ckpt = checkpoint['running_std']
            if rank == 0:
                print(f"Loaded running_std from checkpoint: {running_std_ckpt}")
        else:
            running_std_ckpt = None
            if rank == 0:
                print("Warning: running_std not found in checkpoint. Will use default initialization.")

        del checkpoint

    elif checkpoint_file is not None and rank == 0:
        print(f"Warning: Checkpoint file {checkpoint_file} not found. Starting from scratch.")

        # Load running stats for advantage normalization (with backward compatibility)
        if 'running_mean' in checkpoint:
            running_mean_ckpt = checkpoint['running_mean']
            if rank == 0:
                print(f"Loaded running_mean from checkpoint: {running_mean_ckpt}")
        else:
            running_mean_ckpt = None
            if rank == 0:
                print("Warning: running_mean not found in checkpoint. Will use default initialization.")

        if 'running_std' in checkpoint:
            running_std_ckpt = checkpoint['running_std']
            if rank == 0:
                print(f"Loaded running_std from checkpoint: {running_std_ckpt}")
        else:
            running_std_ckpt = None
            if rank == 0:
                print("Warning: running_std not found in checkpoint. Will use default initialization.")

    # CRITICAL FIX: When starting from scratch (no checkpoint loaded),
    # model_baseline must have the SAME weights as model_train.
    # Otherwise, advantage = kemeny_dist_train - kemeny_dist_baseline will be wrong.
    if checkpoint_file is None or not os.path.exists(checkpoint_file if checkpoint_file else ""):
        model_baseline.load_state_dict(model_train.state_dict())
        if rank == 0:
            print("Initialized model_baseline with model_train weights (no checkpoint loaded).")

    # --- DDP Wrapping ---
    # Now that models are loaded, wrap them in DDP if needed
    if is_ddp:
        model_train = DDP(model_train, device_ids=[rank], find_unused_parameters=True)
        if model_baseline:
            model_baseline = DDP(model_baseline, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        print(f"Model initialized with {sum(p.numel() for p in model_train.parameters())} parameters.")


    # 3. --- Training Loop ---
    epoch_start = epoch_ckpt
    start_training_time_sec = time.time()
    grad_acc_steps = getattr(args, 'gradient_accumulation_steps', None)
    if grad_acc_steps is None:
        grad_acc_steps = 1
    print(f"Gradient Accumulation Steps: {grad_acc_steps}")
    if rank == 0:
        print(f"Gradient Accumulation Steps: {grad_acc_steps}")

    # --- Advantage Normalization Setup ---
    # Options: "running" (EMA-based std scaling) or "batch" (per-batch centering) or "none"
    advantage_norm_type = getattr(args, 'advantage_normalization', None)
    if advantage_norm_type is None:
        advantage_norm_type = "none"  # Default
    advantage_norm_type = advantage_norm_type.lower()
    if rank == 0:
        print(f"Advantage Normalization Type: {advantage_norm_type}")

    # Running stats for EMA-based normalization (used if advantage_norm_type == "running_mean")
    running_mean = 0.0  # Initial running mean
    running_std = 1.0   # Initial running std
    running_stats_momentum = getattr(args, 'running_std_momentum', None)
    if running_stats_momentum is None:
        running_stats_momentum = 0.99  # Default value
    if rank == 0:
        # Override with checkpoint values if loaded
        if 'running_mean_ckpt' in dir() and running_mean_ckpt is not None:
            running_mean = running_mean_ckpt
        if 'running_std_ckpt' in dir() and running_std_ckpt is not None:
            running_std = running_std_ckpt
        print(f"Running Stats Momentum: {running_stats_momentum}")
        print(f"Running Mean: {running_mean}, Running Std: {running_std}")

    sample_items_from_range = getattr(args, 'sample_items_from_range', None)
    sample_voters_from_range = getattr(args, 'sample_voters_from_range', None)
    # Convert string "True"/"False" to boolean (str.lower() handles case-insensitivity)
    if sample_items_from_range is not None:
        sample_items_from_range = str(sample_items_from_range).lower() == 'true'
    if sample_voters_from_range is not None:
        sample_voters_from_range = str(sample_voters_from_range).lower() == 'true'
    for epoch in range(epoch_start, args.nb_epochs):
        if rank == 0:
            print(f"Starting Epoch: {epoch}")
        epoch_start_time_sec = time.time() # For epoch timing
        epoch_start_time_sec = time.time() # For epoch timing
        model_train.train()
        optimizer.zero_grad() # Initialize gradients before loop

        # NOTE: If using a torch.utils.data.DistributedSampler, you would set the epoch here:
        # if is_ddp and train_sampler:
        #     train_sampler.set_epoch(epoch)

        epoch_losses = []
        epoch_advantages = []

        for step in range(1, args.nb_batch_per_epoch + 1):
            # --- Data Generation ---
            # Each rank generates its own batch of data due to rank-seeded 'dsy'
            # Build common kwargs for sampling parameters
            sample_kwargs = {}
            if sample_items_from_range is not None:
                sample_kwargs['sample_items_from_range'] = sample_items_from_range
            if sample_voters_from_range is not None:
                sample_kwargs['sample_voters_from_range'] = sample_voters_from_range

            if args.data_generation_method == "Mallows":
                batch_rankings, _, _ = data_synthesis.batch_generate_base_rankings_Mallows_all_same_shape_vcode(
                    bsz=args.bsz,
                    num_voters_range=args.num_voters_range,
                    num_items_range=args.num_items_range,
                    phi_range=args.phi_range,
                    **sample_kwargs
                )
            elif args.data_generation_method == "random":
                batch_rankings, _ = data_synthesis.generate_batch_dataset_random_from_range(
                    args.bsz,
                    args.num_voters_range,
                    args.num_items_range,
                    **sample_kwargs
                )
            elif args.data_generation_method == "fine_tuning":
                batch_rankings, _ = data_synthesis.generate_batch_instances_fine_tuning(
                    args.bsz,
                    args.num_voters_range,
                    args.num_items_range,
                    **sample_kwargs
                )
            elif args.data_generation_method == "fine_tuning_mix":
                batch_rankings, _ = data_synthesis.generate_mix_batch_instances_fine_tuning(
                    args.bsz,
                    args.num_voters_range,
                    args.num_items_range,
                    **sample_kwargs
                )
            elif args.data_generation_method == "random_mix":
                batch_rankings, _ = data_synthesis.generate_mix_batch_dataset_random_from_range(
                    args.bsz,
                    args.num_voters_range,
                    args.num_items_range,
                    **sample_kwargs
                )
            else:
                raise ValueError(f"Unknown data_generation_method: {args.data_generation_method}")

            if isinstance(batch_rankings, np.ndarray):
                batch_rankings = [batch_rankings[i] for i in range(batch_rankings.shape[0])]
            batch_rankings_token, padding_mask, voter_mask = tokenizer.tokenize(batch_rankings, embedding_type=parsed_embedding_type)
            batch_rankings_token = batch_rankings_token.to(device)
            padding_mask = padding_mask.to(device)
            voter_mask = voter_mask.to(device)
            # --- Policy (Actor) Forward Pass ---
            # Call model_train (which is DDP-wrapped if is_ddp)

            orders_train, sum_log_prob,_ = model_train(x=batch_rankings_token, padding_mask=padding_mask, voter_mask=voter_mask, deterministic=False)

            #print(f"{orders_train[0]=}")
            cleaned_orders_train = clean_padded_permutations(orders_train)
            rankings_train = order_to_rank_batch(cleaned_orders_train)
            #print(f"{rankings_train[0]=}")
            # --- Kemeny Distance Calculation (Actor) ---
            kemeny_dist_train = torch.from_numpy(kemeny_distance_batch(batch_rankings, rankings_train)).to(device, dtype=torch.float32)

            # --- Baseline Calculation ---
            if training_phase == 1:
                print("phase1")
                # PHASE 1: Use kiwi heuristic
                #order_bdks = bdks(batch_rankings)
                #rankings_bdks = order_to_rank_batch(order_bdks)
                #kemeny_dist_baseline = torch.from_numpy(kemeny_distance_batch(batch_rankings, rankings_bdks)).to(device, dtype=torch.float32)
            else:
                # PHASE 2: Use self-improving model_baseline
                with torch.no_grad():
                    orders_baseline, _,_ = model_baseline(x=batch_rankings_token, padding_mask=padding_mask, voter_mask=voter_mask, deterministic=True) # Use deterministic for baseline
                    cleaned_orders_baseline = clean_padded_permutations(orders_baseline)
                    rankings_baseline = order_to_rank_batch(cleaned_orders_baseline)
                    kemeny_dist_baseline = torch.from_numpy(kemeny_distance_batch(batch_rankings, rankings_baseline)).to(device, dtype=torch.float32)

            # --- Loss Calculation & Backpropagation (REINFORCE) ---
            # --- Loss Calculation & Backpropagation (REINFORCE) ---
            advantage = kemeny_dist_train - kemeny_dist_baseline
            #print(f"{kemeny_dist_train=}")
            #print(f"{kemeny_dist_baseline=}")
            #print(f"{advantage=}")

            # >>>> ADVANTAGE NORMALIZATION (configurable) <<<<
            if advantage.shape[0] > 1:
                if advantage_norm_type == "running":
                    # Running mean and std: full standardization with EMA
                    batch_mean_tensor = advantage.mean()
                    batch_std_tensor = advantage.std()

                    # DDP Sync: average stats across all GPUs
                    if is_ddp:
                        dist.all_reduce(batch_mean_tensor, op=dist.ReduceOp.AVG)
                        dist.all_reduce(batch_std_tensor, op=dist.ReduceOp.AVG)

                    batch_mean = batch_mean_tensor.item()
                    batch_std = batch_std_tensor.item()

                    # Update running statistics with EMA
                    running_mean = running_stats_momentum * running_mean + (1 - running_stats_momentum) * batch_mean
                    running_std = running_stats_momentum * running_std + (1 - running_stats_momentum) * batch_std

                    # Normalize using running stats
                    advantage = (advantage - running_mean) / (running_std + 1e-8)
                elif advantage_norm_type == "batch":
                    # Per-batch centering (WARNING: can flip sign)
                    adv_mean = advantage.mean()
                    adv_std = advantage.std()
                    advantage = (advantage - adv_mean) / (adv_std + 1e-8)
                elif advantage_norm_type == "none":
                    # No normalization - use raw advantage
                    pass
                else:
                    # Scale-only (default fallback for unknown types)
                    adv_std = advantage.std()
                    advantage = advantage / (adv_std + 1e-8)

            #print(f"normalized advantage: {advantage}")
            # The loss is automatically averaged across GPUs by DDP during backward()
            #print(f"average advantage: {torch.mean(advantage.detach())}")
            loss = torch.mean(advantage.detach() * sum_log_prob)

            # Scale loss for gradient accumulation
            loss = loss / grad_acc_steps
            loss.backward()

            if (step+1) % grad_acc_steps == 0:
                nn.utils.clip_grad_norm_(model_train.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            # Multiply by grad_acc_steps to log the original scale loss
            epoch_losses.append(loss.item() * grad_acc_steps)
            epoch_advantages.append(torch.mean(advantage).item())

            if rank == 0 and step % 50 == 0:
                current_total_time = (time.time() - start_training_time_sec) + tot_time_ckpt
                # Log original scale loss
                print(f'Epoch:{epoch}/{args.nb_epochs} | Batch: {step}/{args.nb_batch_per_epoch} | Avg Cost (Adv): {np.mean(epoch_advantages):.4f} | Loss:{loss.item() * grad_acc_steps:.4f} | tot_time: {current_total_time/60:.2f}min' )

        avg_epoch_loss = np.mean(epoch_losses)
        if rank == 0:
            print(f"Epoch {epoch}/{args.nb_epochs} | Avg Loss: {avg_epoch_loss:.4f} | Phase: {training_phase}")

        # 4. --- Custom Validation (Rank 0 Only) ---
        if rank == 0:
            # Pass the *unwrapped* model for validation
            model_to_validate = model_train.module if is_ddp else model_train
            print(f"Validation begins")
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

        # 5. --- End-of-Epoch Evaluation and Baseline Update (All Ranks) ---
        model_train.eval()
        if model_baseline:
            model_baseline.eval()

        dist_train_eval_local = np.array([])
        dist_base_eval_local = np.array([])

        with torch.no_grad():
            # world_size is 1 for single-GPU
            eval_bsz_per_gpu = args.roll_out_bsz // world_size
            sample_kwargs = {}
            if sample_items_from_range is not None:
                sample_kwargs['sample_items_from_range'] = sample_items_from_range
            if sample_voters_from_range is not None:
                sample_kwargs['sample_voters_from_range'] = sample_voters_from_range

            if args.data_generation_method == "Mallows":
                eval_raw_rankings, _, _ = data_synthesis.batch_generate_base_rankings_Mallows_all_same_shape_vcode(
                    bsz=eval_bsz_per_gpu,
                    num_voters_range=args.num_voters_range,
                    num_items_range=args.num_items_range,
                    phi_range=args.phi_range,
                    **sample_kwargs
                )
            if args.data_generation_method == "random":
                eval_raw_rankings, is_all_permutation= data_synthesis.generate_batch_dataset_random_from_range(eval_bsz_per_gpu, args.num_voters_range, args.num_items_range, **sample_kwargs)
            if args.data_generation_method == "fine_tuning":
                eval_raw_rankings, _ = data_synthesis.generate_batch_instances_fine_tuning(eval_bsz_per_gpu, args.num_voters_range, args.num_items_range, **sample_kwargs)
            if args.data_generation_method == "fine_tuning_mix":
                eval_raw_rankings, _ = data_synthesis.generate_mix_batch_instances_fine_tuning(eval_bsz_per_gpu, args.num_voters_range, args.num_items_range, **sample_kwargs)
            if args.data_generation_method == "random_mix":
                eval_raw_rankings, _ = data_synthesis.generate_mix_batch_dataset_random_from_range(eval_bsz_per_gpu, args.num_voters_range, args.num_items_range, **sample_kwargs)
            if eval_raw_rankings is None:
                if rank == 0:
                    print(f"Warning: eval_bsz_per_gpu is 0 ({args.roll_out_bsz}/{world_size}). Skipping baseline update evaluation.")
                if is_ddp:
                    dist.barrier() # Ensure all processes continue
                continue # Skip to next epoch

            if isinstance(eval_raw_rankings, np.ndarray):
                eval_raw_rankings = [eval_raw_rankings[i] for i in range(eval_raw_rankings.shape[0])]
            eval_rankings_token, eval_padding_mask, eval_voter_mask = tokenizer.tokenize(eval_raw_rankings, embedding_type=parsed_embedding_type)
            eval_rankings_token = eval_rankings_token.to(device)
            eval_padding_mask = eval_padding_mask.to(device)
            eval_voter_mask = eval_voter_mask.to(device)

            perm_train_eval, _,_ = model_train(x=eval_rankings_token, padding_mask=eval_padding_mask, voter_mask=eval_voter_mask, deterministic=True)
            cleaned_perm_train = clean_padded_permutations(perm_train_eval)
            rank_train_eval = order_to_rank_batch(cleaned_perm_train)
            dist_train_eval_local = kemeny_distance_batch(eval_raw_rankings, rank_train_eval)

            if training_phase == 1:
                print(f"pahse1")
                #order_base_eval_np = bdks(eval_raw_rankings)
                #rank_base_eval_np = order_to_rank_batch(order_base_eval_np)
                #dist_base_eval_local = kemeny_distance_batch(eval_raw_rankings, rank_base_eval_np)
            else:
                perm_base_eval, _,_ = model_baseline(x=eval_rankings_token, padding_mask=eval_padding_mask, voter_mask=eval_voter_mask, deterministic=True)
                cleaned_perm_base = clean_padded_permutations(perm_base_eval)
                rank_base_eval = order_to_rank_batch(cleaned_perm_base)
                dist_base_eval_local = kemeny_distance_batch(eval_raw_rankings, rank_base_eval)

            # Gather results from all GPUs if DDP is enabled
            if is_ddp:
                dist_train_eval_gathered = [np.zeros_like(dist_train_eval_local) for _ in range(world_size)]
                dist_base_eval_gathered = [np.zeros_like(dist_base_eval_local) for _ in range(world_size)]
                dist.all_gather_object(dist_train_eval_gathered, dist_train_eval_local)
                dist.all_gather_object(dist_base_eval_gathered, dist_base_eval_local)
            else:
                # In single-GPU mode, the "gathered" list just contains the local results
                dist_train_eval_gathered = [dist_train_eval_local]
                dist_base_eval_gathered = [dist_base_eval_local]

            update_baseline_flag = torch.tensor(0, device=device) # Flag for non-rank0

            if rank == 0:
                dist_train_eval = np.concatenate(dist_train_eval_gathered)
                dist_base_eval = np.concatenate(dist_base_eval_gathered)

                mean_dist_train = np.mean(dist_train_eval)
                mean_dist_base = np.mean(dist_base_eval)

                baseline_name = "kiwi" if training_phase == 1 else "Model Baseline"
                print(f"  Baseline Eval: Train Model Avg Kemeny: {mean_dist_train:.4f} | {baseline_name} Avg Kemeny: {mean_dist_base:.4f}")

                update_baseline = False
                if mean_dist_train < mean_dist_base:
                    try:
                        t_stat, p_value = stats.ttest_rel(dist_train_eval, dist_base_eval)
                        if t_stat < 0 and p_value / 2 < args.baseline_alpha:
                            print(f"  > Statistically significant improvement found (p={p_value/2:.5f}).")
                            update_baseline = True
                        else:
                            print(f"  > Improvement not statistically significant (p={p_value/2:.5f}).")
                    except ValueError as e:
                        print(f"  > T-test failed (dist_train={dist_train_eval.shape}, dist_base={dist_base_eval.shape}): {e}")

                #  Write to Rollout Log ---
                if rollout_log_handle:
                    time_one_epoch = time.time() - epoch_start_time_sec
                    time_tot = (time.time() - start_training_time_sec) + tot_time_ckpt
                    rollout_log_record = (
                        f'Epoch: {epoch}, epoch time: {time_one_epoch/60:.2f}min, tot time: {time_tot/86400:.2f}day, '
                        f'kemeny_dis_train: {mean_dist_train:.6f}, kemeny_dis_base: {mean_dist_base:.6f}, '
                        f'update: {update_baseline}, training_phase: {training_phase}'
                    )
                    rollout_log_handle.write(rollout_log_record + '\n')
                    rollout_log_handle.flush()
                # --- End of New Log Block ---

                if update_baseline:
                    update_baseline_flag.fill_(1) # Set flag to broadcast

                    # Get the state dict of the *unwrapped* training model
                    train_model_state_dict = model_train.module.state_dict() if is_ddp else model_train.state_dict()

                    if training_phase == 1:
                        print("\n" + "="*50)
                        print("PHASE 1 COMPLETE: Model has surpassed the Derkiwi heuristic.")
                        print("SWITCHING TO PHASE 2: Self-improvement.")
                        print("="*50 + "\n")
                        training_phase = 2

                        # Initialize baseline model
                        model_baseline = kemeny_transformer(
                            embedding_type=parsed_embedding_type,
                            input_dim=args.dim_input,
                            embedding_dim=args.dim_emb,
                            dim_ff=args.dim_ff,
                            numb_heads=args.numb_heads,
                            numb_layers_decoder=args.numb_layers_decoder,
                            numb_layers_encoder=args.numb_layers_encoder,
                            max_len_PE=args.max_len_PE,
                            conv_out_channels = args.conv_out_channels,
                            batchnorm= args.batchnorm,
                        ).to(device)
                        model_baseline.load_state_dict(train_model_state_dict)
                        # Wrap in DDP if needed
                        if is_ddp:
                            model_baseline = DDP(model_baseline, device_ids=[rank], find_unused_parameters=True)

                    elif training_phase == 2:
                        print("  > Updating model baseline with new weights.")
                        # Load state dict into the *unwrapped* baseline model
                        if is_ddp:
                            model_baseline.module.load_state_dict(train_model_state_dict)
                        else:
                            model_baseline.load_state_dict(train_model_state_dict)

            # --- DDP-specific Syncing Block ---
            if is_ddp:
                # Broadcast the training phase and update flag to all processes
                training_phase_tensor = torch.tensor(training_phase, device=device)
                dist.broadcast(training_phase_tensor, src=0)
                dist.broadcast(update_baseline_flag, src=0)

                training_phase = training_phase_tensor.item()

                dist.barrier()

                # If phase 2 just started, or baseline was updated, non-rank-0 processes must initialize/update their baseline model
                if training_phase == 2:
                    if model_baseline is None:
                        # This is the first time we enter phase 2 for this non-rank0 process
                        if rank != 0:
                            print(f"Rank {rank} initializing Phase 2 baseline model.")
                        model_baseline = kemeny_transformer(
                            embedding_type=parsed_embedding_type,
                            input_dim=args.dim_input,
                            embedding_dim=args.dim_emb,
                            dim_ff=args.dim_ff,
                            numb_heads=args.numb_heads,
                            numb_layers_decoder=args.numb_layers_decoder,
                            numb_layers_encoder=args.numb_layers_encoder,
                            max_len_PE=args.max_len_PE,
                            conv_out_channels = args.conv_out_channels,
                            batchnorm= args.batchnorm,
                        ).to(device)
                        # Non-rank 0s load from model_train, which was synced by DDP
                        model_baseline.load_state_dict(model_train.module.state_dict())
                        model_baseline = DDP(model_baseline, device_ids=[rank], find_unused_parameters=True)

                    elif update_baseline_flag.item() == 1 and rank != 0:
                        # Baseline was updated on rank 0, sync weights
                        print(f"Rank {rank} syncing baseline model weights.")
                        model_baseline.module.load_state_dict(model_train.module.state_dict())
            # --- End DDP-specific Syncing Block ---


        # --- Checkpointing (Rank 0 Only) ---
        checkpoint_dir = getattr(args, 'checkpoint_dir', 'checkpoints')
        save_every = getattr(args, 'save_every', 5)
        if rank == 0 and epoch % save_every == 0 and epoch > 0:
            current_total_time = (time.time() - start_training_time_sec) + tot_time_ckpt

            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)

            save_path = f'checkpoint_epoch_{epoch}.pkl'
            if checkpoint_dir:
                save_path = os.path.join(checkpoint_dir, save_path)

            # Save the *unwrapped* state dicts
            train_state = model_train.module.state_dict() if is_ddp else model_train.state_dict()
            base_state = None
            if model_baseline:
                base_state = model_baseline.module.state_dict() if is_ddp else model_baseline.state_dict()

            torch.save({
                'epoch': epoch,
                'model_train_state_dict': train_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'training_phase': training_phase,
                'model_baseline_state_dict': base_state,
                'tot_time': current_total_time
            }, save_path)
            print(f"  > Checkpoint saved to {save_path}.")

        if is_ddp:
            dist.barrier() # Sync all processes before next epoch

        # Step the scheduler at the end of epoch
        scheduler.step()

    if rank == 0:
        print("Training complete.")
        if validation_log_handle:
            print("Closing validation log file.")
            validation_log_handle.close()
        if rollout_log_handle:
            print("Closing rollout log file.")
            rollout_log_handle.close()

    if is_ddp:
        ddp_cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Kemeny Transformer Training")
    parser.add_argument('--config_file', type=str, required=True, help='Path to the JSON configuration file.')
    cli_args = parser.parse_args()

    # Load configuration from JSON file
    try:
        with open(cli_args.config_file, 'r') as f:
            config_args_dict = json.load(f)

        # Convert config dict to an object (like argparse.Namespace)
        # so train() can access args with dot notation (e.g., args.lr)
        class ConfigObject:
            def __init__(self, **entries):
                self.__dict__.update(entries)

            def __getattr__(self, name):
                # Return None if arg is not in config, instead of raising error
                return self.__dict__.get(name, None)

        args_obj = ConfigObject(**config_args_dict)

        # Removed the 'LOCAL_RANK' check, as the script now handles
        # both 'torchrun' and 'python' execution.

        train(args_obj)

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {cli_args.config_file}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {cli_args.config_file}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Clean up DDP in case it was initialized before the error
        if dist.is_initialized():
            ddp_cleanup()
        exit(1)
