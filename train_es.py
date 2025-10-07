"""
Minimal ES (Evolution Strategies) implementation for LLM fine-tuning.
Implements Algorithm 2 from the paper.
"""
import json
import math
import random
import re
from fractions import Fraction
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM, GenerationConfig
import wandb


# system prompt for math tasks
system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>
"""


_ANS_RX = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL | re.IGNORECASE)

def _norm(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")                      # 1,000 -> 1000
    s = s.replace("$", "")                      # strip LaTeX $...$
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s) # \boxed{42} -> 42
    s = re.sub(r"\s+", " ", s)                  # collapse whitespace
    return s

def _parse_num(s: str):
    # simple numeric forms: int / float / fraction
    if re.fullmatch(r"[+-]?\d+/\d+", s):
        try: return float(Fraction(s))
        except Exception: return None
    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", s):
        try: return float(s)
        except Exception: return None
    return None

def score_answer_simple(completion: str, oracle_answer: str, rel=1e-9, abs_=1e-9, partial_credit=True):
    m = _ANS_RX.search(completion)
    if not m:
        return 0.0
    pred, gold = _norm(m.group(1)), _norm(oracle_answer)

    pn, gn = _parse_num(pred), _parse_num(gold)
    if pn is not None and gn is not None:
        return 1.0 if math.isclose(pn, gn, rel_tol=rel, abs_tol=abs_) else 0.01

    if pred == gold:
        return 1.0

    if partial_credit:
        # boundary-aware (avoid "12" inside "312")
        if re.search(rf"(?<!\w){re.escape(gold)}(?!\w)", pred):
            return 0.5

    return 0.01


def load_model(model_name: str, device):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    return model, tokenizer


@torch.no_grad()
def compute_reward(model, tokenizer, question: str, oracle_answer: str, max_length: int = 1024):
    """Generate completion and compute reward. Returns (reward, completion)."""
    model.eval()
    
    # format prompt
    chat_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )
    
    # generate (greedy decoding for deterministic evaluation)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)
    generation_config = GenerationConfig(
        do_sample=False,  # greedy decoding
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = model.generate(**inputs, generation_config=generation_config)
    completion = tokenizer.decode(
        output_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    
    # compute reward using robust scoring
    reward = score_answer_simple(completion, oracle_answer, partial_credit=True)
    return reward, completion


@torch.no_grad()
def compute_reward_batch(model, tokenizer, questions: list[str], oracle_answers: list[str], max_length: int = 1024):
    """Generate completions for batch of questions and compute rewards. Returns (rewards, completions)."""
    model.eval()
    
    # format prompts
    batch_prompts = []
    for question in questions:
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
        batch_prompts.append(prompt)
    
    # batch tokenize
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(model.device)
    
    # generate for all questions at once
    generation_config = GenerationConfig(
        do_sample=False,  # greedy decoding
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
    )
    output_ids = model.generate(**inputs, generation_config=generation_config)
    
    # decode completions
    input_length = inputs["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        output_ids[:, input_length:], skip_special_tokens=True
    )
    
    # compute rewards
    rewards = []
    for completion, oracle_answer in zip(completions, oracle_answers):
        reward = score_answer_simple(completion, oracle_answer, partial_credit=True)
        rewards.append(reward)
    
    return rewards, completions


def generate_noise(model, seed: int):
    """Generate and return noise vectors for all parameters."""
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    noise_vectors = []
    for param in model.parameters():
        noise = torch.randn(param.shape, generator=generator, dtype=param.dtype, device=param.device)
        noise_vectors.append(noise)
    return noise_vectors


def apply_noise(model, noise_vectors: list, scale: float):
    """Apply noise vectors to model parameters."""
    for param, noise in zip(model.parameters(), noise_vectors):
        param.data.add_(noise, alpha=scale)


def train_es(
    model,
    tokenizer,
    questions: list[str],
    answers: list[str],
    N: int = 12,
    batch_size: int = 32,
    sigma: float = 0.01,
    alpha: float = 0.001,
    T: int = 100,
    max_length: int = 1024,
):
    """
    Batched ES training loop with seed regeneration and batched generation.
    
    Args:
        model: LLM to optimize
        tokenizer: tokenizer
        questions: list of training questions
        answers: list of oracle answers
        N: population size (number of perturbations)
        batch_size: number of questions per iteration (B)
        sigma: noise scale
        alpha: learning rate
        T: number of iterations
        max_length: max generation length
    """
    
    ema_reward = None
    
    for t in range(T):
        print(f"\n=== Iteration {t+1}/{T} ===")
        
        # sample batch_size random questions
        indices = random.sample(range(len(questions)), batch_size)
        batch_questions = [questions[i] for i in indices]
        batch_answers = [answers[i] for i in indices]
        
        # initialize gradient accumulator
        gradient_accum = [torch.zeros_like(p) for p in model.parameters()]
        
        # generate N random seeds (one per perturbation)
        seeds = [random.randint(0, 2**32 - 1) for _ in range(N)]
        
        # track rewards: questions x perturbations
        question_rewards = [[] for _ in range(batch_size)]
        question_completions = [[] for _ in range(batch_size)]
        
        # for each perturbation, evaluate ALL questions (batched)
        print(f"  evaluating {N} perturbations on {batch_size} questions...")
        for n in range(N):
            # generate and apply noise
            noise_vectors = generate_noise(model, seeds[n])
            apply_noise(model, noise_vectors, sigma)
            
            # evaluate ALL questions on this perturbed model (batched generation)
            rewards, completions = compute_reward_batch(
                model, tokenizer, batch_questions, batch_answers, max_length
            )
            
            # store rewards and completions per question
            for q_idx in range(batch_size):
                question_rewards[q_idx].append(rewards[q_idx])
                question_completions[q_idx].append(completions[q_idx])
            
            # restore model
            apply_noise(model, noise_vectors, -sigma)
            
            if (n + 1) % 4 == 0 or n == N - 1:
                print(f"    completed perturbation {n+1}/{N}")
        
        # normalize per question and collect z_scores
        all_rewards = []
        all_z_scores = []  # batch_size x N
        
        for q_idx, (question, answer) in enumerate(zip(batch_questions, batch_answers)):
            rewards = question_rewards[q_idx]
            completions = question_completions[q_idx]
            
            # z-score normalize within this question's group
            R_mean = sum(rewards) / len(rewards)
            R_std = (sum((r - R_mean)**2 for r in rewards) / len(rewards)) ** 0.5
            R_std = max(R_std, 1e-8)
            
            z_scores = [(r - R_mean) / R_std for r in rewards]
            all_z_scores.append(z_scores)
            
            # track stats
            all_rewards.extend(rewards)
            
            # print stats for this question
            best_idx = rewards.index(max(rewards))
            print(f"  Q{q_idx+1}: '{question[:50]}...' -> '{answer}'")
            print(f"    mean={R_mean:.4f}, std={R_std:.4f}, max={max(rewards):.4f}")
            print(f"    best: {completions[best_idx][:120]}...")
        
        # accumulate gradient by regenerating noise once per perturbation
        print(f"  computing gradient...")
        for n in range(N):
            noise_vectors = generate_noise(model, seeds[n])
            # apply each question's z-score independently
            for q_idx in range(batch_size):
                z_score = all_z_scores[q_idx][n]
                for grad, noise in zip(gradient_accum, noise_vectors):
                    # standard ES update: z_i * ε_i / (N), averaged over batch
                    grad.add_(noise, alpha=z_score / (N *  batch_size))
        
        # apply accumulated gradient
        for param, grad in zip(model.parameters(), gradient_accum):
            param.data.add_(grad, alpha=alpha)
        
        print(f"  model updated")
        
        # compute batch-level stats
        overall_mean = sum(all_rewards) / len(all_rewards)
        overall_std = (sum((r - overall_mean)**2 for r in all_rewards) / len(all_rewards)) ** 0.5
        
        # update ema
        if ema_reward is None:
            ema_reward = overall_mean
        else:
            ema_reward = 0.9 * ema_reward + 0.1 * overall_mean
        
        print(f"  batch stats: reward_mean={overall_mean:.4f}, reward_std={overall_std:.4f}, reward_ema={ema_reward:.4f}")
        
        # log to wandb
        wandb.log({
            "iteration": t + 1,
            "reward_mean": overall_mean,
            "reward_std": overall_std,
            "reward_max": max(all_rewards),
            "reward_min": min(all_rewards),
            "reward_ema": ema_reward,
        }, step=t + 1)


def read_jsonl(file_name: str | Path):
    """Read jsonl file."""
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate = None,
    max_rows = None,
) -> list:
    """Read and filter prompts from jsonl file."""
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows


def main():
    # hyperparameters
    seed = 42
    device = torch.device("cuda:0")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    N = 30  # population size (perturbations per question)
    batch_size = 8  # questions per iteration
    sigma = 0.001  # noise scale
    alpha = 0.0005  # learning rate
    T = 10000  # iterations
    max_length = 1024
    wandb_project = "tiny-grpo-es"
    
    # set seed
    random.seed(seed)
    torch.manual_seed(seed)
    
    # initialize wandb
    wandb.init(
        project=wandb_project,
        config={
            "seed": seed,
            "model_name": model_name,
            "N": N,
            "batch_size": batch_size,
            "sigma": sigma,
            "alpha": alpha,
            "T": T,
            "max_length": max_length,
        }
    )
    
    # load model
    print("loading model...")
    model, tokenizer = load_model(model_name, device)
    
    # load data
    print("loading data...")
    prompts = read_prompts(
        "data/math_tasks.jsonl",
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    
    questions = [p["question"] for p in prompts]
    answers = [p["answer"] for p in prompts]
    print(f"loaded {len(questions)} questions")
    
    # train
    print("\nstarting ES training...")
    train_es(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        answers=answers,
        N=N,
        batch_size=batch_size,
        sigma=sigma,
        alpha=alpha,
        T=T,
        max_length=max_length,
    )
    
    # save
    output_path = Path("./output/es_final")
    model.save_pretrained(output_path)
    print(f"\nmodel saved to {output_path}")


if __name__ == "__main__":
    main()

