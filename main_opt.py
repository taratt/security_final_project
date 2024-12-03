import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.attack_opt import perform_attack
from lib.eval import eval_ppl

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='OPT model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the output model.')


    parser.add_argument("--bfa", action="store_true", help='Perform bfa on the model')
    parser.add_argument("--bfa_percentage", type=float, default=1, help='Per layer bitflip percentage')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    print("use device ", device)

    if args.bfa:
        print("Attack starts")
        perform_attack(args, model, tokenizer, device, args.bfa_percentage)

    ################################################################
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.bfa}.txt")
    with open(save_filepath, "w") as f:
        print("method\tbfa\tbfa_percentage\tppl_test", file=f, flush=True)
        print(f"{args.bfa}\t{args.bfa_percentage:.4f}\t{ppl_test:.4f}", file=f, flush=True)


    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()