import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
import logging as pylog
import numpy as np
import os
from accelerate import Accelerator
import time
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import gc
import wandb
from src.config import Config

from src.rewards import Score
try:
    from src.rewards.rewards import set_reward_iter  # local, optional
except Exception:
    def set_reward_iter(step):
        return None

logging.set_verbosity_error()
torch.backends.cuda.matmul.allow_tf32 = True
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
    # Disable flash/mem-efficient SDPA to use a more robust attention path
    torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
# Load project configuration (env/.env backed)
config = Config()

# Configure Python logging so reward breakdown INFO logs are visible
try:
    level_name = str(os.getenv("LOG_LEVEL", "INFO")).upper()
    level = getattr(pylog, level_name, pylog.INFO)
    if not pylog.getLogger().handlers:
        pylog.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        pylog.getLogger().setLevel(level)
except Exception:
    pass


# Hyperparameters for ES (use sensible defaults; could be added to Config later)
NUM_ITERATIONS = 1000            # iterations (generations)
POPULATION_SIZE = 50             # perturbations per iteration
SIGMA = 0.001                    # noise scale
ALPHA = 0.0005                   # learning rate
MAX_NEW_TOKENS = config.max_new_tokens
DO_SAMPLE = False                # greedy decoding for ES evaluation
INITIAL_SEED = 33


# --- Prompts to evaluate ---
# For plasmid design we optimize the model to directly output a DNA sequence. Use a
# single default prompt from Config and evaluate the score of the completion.
DATASET_PROMPTS = [config.default_query]

def force_memory_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

def save_model_checkpoint(model, tokenizer, iteration, model_name, initial_seed, dataset_size, precision: str, gpu_threads: int):
    """Save model checkpoint at specified iteration"""
    question_num = dataset_size
    save_dir = f"{model_name}_es_random_seed{initial_seed}_pop{POPULATION_SIZE}_iter{iteration}_sigma{SIGMA}_alpha{ALPHA}_{precision}_threads{gpu_threads}_question_num{question_num}_checkpoint"
    print(f"Saving checkpoint at iteration {iteration} to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Checkpoint saved successfully.")

def evaluate_model(model, tokenizer, input_texts, accelerator, seed_idx=None, thread_id=None, verbose=False, return_text=False, return_breakdown=False):
    """
    Generate responses from the model for a batch of prompts and compute rewards using Score.
    """
    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} evaluating seed {seed_idx}")

    # Batch tokenization with safe truncation to keep within context window
    max_ctx = int(getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024)))
    safe_input_len = max(1, max_ctx - int(MAX_NEW_TOKENS))
    tokenized_inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=safe_input_len,
    )
    input_ids = tokenized_inputs["input_ids"].to(accelerator.device)
    attention_mask = tokenized_inputs["attention_mask"].to(accelerator.device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

    # Decode batch outputs
    generated_texts = []
    for i in range(len(input_texts)):
        try:
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
        except TypeError:
            tokens = tokenizer.convert_ids_to_tokens(outputs[i], skip_special_tokens=True)
            filtered = [t for t in tokens if t is not None]
            generated_text = tokenizer.convert_tokens_to_string(filtered)
        generated_texts.append(generated_text)

    del input_ids, outputs
    torch.cuda.empty_cache()

    # Compute rewards for batch texts using our plasmid Score function
    if return_breakdown:
        rewards, breakdowns = Score(generated_texts, return_breakdown=True)
    else:
        rewards = Score(generated_texts)
        breakdowns = None

    if return_text and return_breakdown:
        return rewards, generated_texts, breakdowns
    elif return_text:
        return rewards, generated_texts
    elif return_breakdown:
        return rewards, breakdowns
    else:
        return rewards

def process_seed(seed_args):
    """Function to process a single seed, used for thread pool"""
    seed_idx, seed, model, tokenizer, accelerator, thread_id, verbose, return_breakdown = seed_args

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} processing seed {seed_idx} (value: {seed})")

    # Put load-evaluate-restore in the same lock block for thread safety
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(SIGMA * noise)

    # Ensure weights are fully loaded before evaluation
    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    # Evaluate all prompts with perturbed weights in batch
    input_texts = list(DATASET_PROMPTS)
    if return_breakdown:
        rewards, breakdowns = evaluate_model(model, tokenizer, input_texts, accelerator,
                               seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, 
                               return_text=False, return_breakdown=True)
    else:
        rewards = evaluate_model(model, tokenizer, input_texts, accelerator,
                               seed_idx=seed_idx, thread_id=thread_id, verbose=verbose, return_text=False)
        breakdowns = None
    
    total_reward = sum(rewards)

    # Restore original weights (direct inplace modification)
    for name, param in model.named_parameters():
        gen = torch.Generator(device=param.device)

        gen.manual_seed(int(seed))

        noise = torch.randn(
            param.shape,
            generator=gen,
            device=param.device,
            dtype=param.dtype
        )
        param.data.add_(-SIGMA * noise)

    if torch.cuda.is_available():
        torch.cuda.synchronize(accelerator.device)

    average_reward = total_reward / max(1, len(DATASET_PROMPTS))

    # Removed per-seed cleanup for speed - cleanup happens at batch level instead

    if verbose:
        print(f"Process {accelerator.process_index} Thread {thread_id} completed seed {seed_idx} with reward {average_reward:.4f}")

    if return_breakdown:
        return seed_idx, average_reward, breakdowns
    return seed_idx, average_reward


# --- Main Evolution Strategies Loop ---
def main():
    accelerator = Accelerator()

    if accelerator.is_main_process:
        print(f"Total processes: {accelerator.num_processes}")
        print(f"Population size: {POPULATION_SIZE}, Iterations: {NUM_ITERATIONS}")
        print(f"Sigma: {SIGMA}, Alpha: {ALPHA}")

    # Load model
    model_name = config.model

    if accelerator.is_main_process:
        print(f"Loading model {model_name}...")

    # Initialize Weights & Biases on main process only
    if accelerator.is_main_process:
        run = wandb.init(
                project=getattr(config, "wandb_project", "plasmidES"),
                entity=getattr(config, "wandb_entity", None),
                name=getattr(config, "wandb_run_name", None) or None,
                tags=getattr(config, "wandb_tags", None),
                notes=getattr(config, "wandb_notes", None),
                config={
                    "model": model_name,
                    "population_size": POPULATION_SIZE,
                    "num_iterations": NUM_ITERATIONS,
                    "sigma": SIGMA,
                    "alpha": ALPHA,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "do_sample": DO_SAMPLE,
                    "initial_seed": INITIAL_SEED,
                },
                reinit=True,
            )
            # Ensure nice x-axis
        wandb.define_metric("iter")
        wandb.define_metric("reward/*", step_metric="iter")
        wandb.define_metric("iter/time_s", step_metric="iter")



    # Load model on main process first then sync
    model_list = []
    GPU_THREADS = 8  # Increased from 4 - more parallel seed evaluations (tune based on GPU memory)
    for model_index in range(GPU_THREADS):
        model_list.append(AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": accelerator.process_index},  # Assign devices explicitly
            torch_dtype=torch.bfloat16,
        ))
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Ensure model(s) use safe generation defaults and robust attention implementation
    for m in model_list:
        # Generation/padding configuration
        m.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(m, "generation_config") and m.generation_config is not None:
            m.generation_config.pad_token_id = tokenizer.pad_token_id
            m.generation_config.eos_token_id = tokenizer.eos_token_id
            m.generation_config.do_sample = False
        if hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = "eager"

    if accelerator.is_main_process:
        print("Model loaded successfully")

    # Prepare model with accelerator
    for model in model_list:
        model.eval()  # Turn off dropout, etc.

    force_memory_cleanup()

    # Record total training start time
    training_start_time = time.time()

    np.random.seed(INITIAL_SEED)

    for iteration in range(NUM_ITERATIONS):
        # Record iteration start time
        iter_start_time = time.time()

        # Force garbage collection
        force_memory_cleanup()

        # verbose logging controlled by env LOG_LEVEL; keep minimal prints here

        # Set wandb step for reward breakdown logs early in the iteration
        set_reward_iter(iteration + 1)

        # Generate seeds on main process only
        if accelerator.is_main_process:
            seeds = np.random.randint(0, 2**30, size=POPULATION_SIZE, dtype=np.int64).tolist()
            seeds_tensor = torch.tensor(seeds, device=accelerator.device)
        else:
            seeds_tensor = torch.zeros(POPULATION_SIZE, dtype=torch.long, device=accelerator.device)

        # Broadcast seeds from main process to all processes
        if accelerator.num_processes>1:
            torch.distributed.broadcast(seeds_tensor, src=0)
        seeds = seeds_tensor.cpu().tolist()  # Convert back to list for all processes

        # seeds received

        # Assign seeds to each process for processing
        local_seeds = []
        for seed_idx, seed in enumerate(seeds):
            # Simple task assignment: assign seeds by process ID
            if seed_idx % accelerator.num_processes == accelerator.process_index:
                local_seeds.append((seed_idx, seed))

        # seeds assigned locally

        # Process seeds in smaller batches to reduce memory pressure
        local_rewards = []
        local_breakdowns = []  # collect breakdowns for averaging
        batch_size = max(1, min(GPU_THREADS, len(local_seeds)))

        for batch_start in range(0, len(local_seeds), batch_size):
            batch_end = min(batch_start + batch_size, len(local_seeds))
            batch_seeds = local_seeds[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=len(batch_seeds)) as executor:
                # Prepare thread arguments - enable breakdown collection
                thread_args = []
                for thread_id, (seed_idx, seed) in enumerate(batch_seeds):
                    thread_args.append((seed_idx, seed, model_list[thread_id], tokenizer, accelerator, thread_id, False, True))

                # Execute in parallel and collect results
                results = list(executor.map(process_seed, thread_args))
                for result in results:
                    seed_idx, reward, breakdowns = result
                    local_rewards.append((seed_idx, reward))
                    local_breakdowns.extend(breakdowns)  # extend with all breakdowns from this seed

            # Clean up between batches (less aggressive to avoid overhead)
            if batch_start % (batch_size * 4) == 0:  # Only cleanup every 4 batches
                force_memory_cleanup()

        # Collect rewards from all processes
        all_rewards = torch.zeros(POPULATION_SIZE, device=accelerator.device)

        # Fill in locally computed rewards
        for seed_idx, reward in local_rewards:
            all_rewards[seed_idx] = reward

        # Aggregate rewards from all processes (each process will get the full reward list)
        if accelerator.num_processes>1:
            torch.distributed.all_reduce(all_rewards, op=torch.distributed.ReduceOp.SUM)

        # Convert aggregated rewards back to Python list
        rewards = all_rewards.cpu().tolist()
        # Clean up no longer needed tensor
        del all_rewards
        force_memory_cleanup()

        # Convert rewards to a tensor and normalize.
        rewards_tensor = np.array(rewards, dtype=np.float32)
        rewards_normalized = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)


        # update model weights
        original_model = model_list[0]
        for name, param in original_model.named_parameters():
            gen = torch.Generator(device=param.device)
            update = torch.zeros_like(param)
            for seed_idx in range(POPULATION_SIZE):
                r_norm = rewards_normalized[seed_idx]
                seed = seeds[seed_idx]
                gen.manual_seed(int(seed))

                noise = torch.randn(
                    param.shape,
                    generator=gen,
                    device=param.device,
                    dtype=param.dtype
                )
                noise.mul_(float(r_norm))
                update.add_(noise)
                del noise
            update.div_(POPULATION_SIZE)
            param.data.add_(ALPHA * update)
            torch.cuda.empty_cache()

        for model_idx in range(1, len(model_list)):
            original_model_tmp = model_list[model_idx]
            for name, param in original_model_tmp.named_parameters():
                param.data.copy_(original_model.get_parameter(name).data.clone())

        # Synchronize to ensure weight updates are complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(accelerator.device)

        force_memory_cleanup()

        iter_time = time.time() - iter_start_time

        mean_reward = rewards_tensor.mean().item()
        min_reward = rewards_tensor.min().item()
        max_reward = rewards_tensor.max().item()

        # Calculate average breakdown across all evaluations this iteration
        avg_breakdown = {}
        if local_breakdowns:
            # Average each component across all breakdowns
            breakdown_keys = local_breakdowns[0].keys()
            for key in breakdown_keys:
                values = [bd.get(key, 0.0) for bd in local_breakdowns]
                avg_breakdown[key] = sum(values) / len(values)

        del rewards_tensor, rewards_normalized
        force_memory_cleanup()

        if accelerator.is_main_process:
            print(f"Iteration {iteration + 1}/{NUM_ITERATIONS}, Time: {iter_time:.2f}s, Mean: {mean_reward:.2f}, Min: {min_reward:.2f}, Max: {max_reward:.2f}")
            print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB allocated, {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB peak")

            # Log metrics to Weights & Biases
            if wandb is not None:
                try:
                    # align reward breakdown logs to the same step
                    set_reward_iter(iteration + 1)
                    log_dict = {
                        "iter": iteration + 1,
                        "reward/mean": mean_reward,
                        "reward/min": min_reward,
                        "reward/max": max_reward,
                        "iter/time_s": iter_time,
                        "gpu/mem_alloc_mb": float(torch.cuda.memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
                        "gpu/mem_peak_mb": float(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else 0.0,
                    }
                    # Add averaged breakdown metrics
                    if avg_breakdown:
                        log_dict["reward/parts/ori"] = avg_breakdown.get("ori", 0.0)
                        log_dict["reward/parts/cassettes"] = avg_breakdown.get("cassettes", 0.0)
                        log_dict["reward/parts/promoters"] = avg_breakdown.get("standalone_promoters", 0.0)
                        log_dict["reward/parts/terminators"] = avg_breakdown.get("standalone_terminators", 0.0)
                        log_dict["reward/parts/payload"] = avg_breakdown.get("payload", 0.0)
                        log_dict["reward/parts/length_penalty"] = avg_breakdown.get("length_penalty", 0.0)
                        log_dict["reward/L"] = avg_breakdown.get("L", 0.0)
                    
                    wandb.log(log_dict)
                except Exception:
                    pass

            # Save checkpoint every 20 iterations
            if (iteration + 1) % 20 == 0:
                save_model_checkpoint(
                    original_model,
                    tokenizer,
                    iteration + 1,
                    model_name,
                    INITIAL_SEED,
                    len(DATASET_PROMPTS),
                    precision="bf16",
                    gpu_threads=GPU_THREADS,
                )

    total_time = time.time() - training_start_time


    # Save the final fine-tuned model weights.
    if accelerator.is_main_process:
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")
        question_num = len(DATASET_PROMPTS)
        save_dir = f"{model_name}_es_random_seed{INITIAL_SEED}_pop{POPULATION_SIZE}_iter{NUM_ITERATIONS}_sigma{SIGMA}_alpha{ALPHA}_bf16_threads{GPU_THREADS}_question_num{question_num}_final"
        print(f"Saving final model to {save_dir}...")
        original_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Final model saved successfully.")
        if wandb is not None:
            try:
                wandb.summary["total_time_s"] = float(total_time)
                wandb.finish()
            except Exception:
                pass

if __name__ == "__main__":
    os.environ["PYTHONWARNINGS"] = "ignore"
    mp.set_start_method('spawn', force=True)
    main()