import argparse

import torch

from tqdm import tqdm

import numpy as np

from transformers import GPT2Tokenizer

from src.models.ssm_seq import SSMLMHeadModel


parser = argparse.ArgumentParser(description='Basic ICL Test')
parser.add_argument('--dmodel', type=int, default=2048)
parser.add_argument('--nlayer', type=int, default=24)
parser.add_argument('--attn-layer-idx', nargs='+', type=int, default=[8,16])
parser.add_argument('--rotary_emb_dim', type=int, default=None, help='For rotary embeddings, set to 64. Default is None.')
parser.add_argument('--nheads', type=int, default=16)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--examples', type=str, help="Path to text corpus with examples for ICL.")
parser.add_argument('--labels', type=str,  help="Path to text corpus with labels for ICL.")
parser.add_argument('--separator', type=str, default=" is ", help="ICL separator between each example and label value.")
parser.add_argument('--query', type=str, default="What is <QUERY>?")
parser.add_argument('--size', type=int, default = 3, help="Size of ICL shot learning")
parser.add_argument('--iters', type=int, default = 1, help ="Number of times to run ICL")
parser.add_argument('--permute_count', type=int, default = 1, help ="Number of permutations to run")
parser.add_argument('--out_path', type=str, help ="Where to save ICL dataset")
args = parser.parse_args()

device = 'cuda'
dtype = torch.float16
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# set seed
torch.random.manual_seed(0)
d_model = args.dmodel
n_layer = args.nlayer
ssm_cfg = dict(mode='diag', measure='diag-lin')
attn_layer_idx = args.attn_layer_idx
if args.rotary_emb_dim is None:
    attn_cfg = dict(num_heads=args.nheads)
else:
    attn_cfg = dict(num_heads=args.nheads, rotary_emb_dim=args.rotary_emb_dim)
model = SSMLMHeadModel(d_model, n_layer=n_layer, d_inner=4 * d_model, vocab_size=len(tokenizer),
                       ssm_cfg=ssm_cfg, attn_layer_idx=attn_layer_idx, attn_cfg=attn_cfg,
                       pad_vocab_size_multiple=8).to(device=device)
if args.ckpt is not None:
    state_dict = torch.load(args.ckpt, map_location=device)
    if 'pytorch-lightning_version' in state_dict:
        state_dict = {k[len('model.'):]: v for k, v in state_dict['state_dict'].items()
                      if k.startswith('model.')}
    model.load_state_dict(state_dict)
model.eval()
# Only cast the nn.Linear parameters to dtype, the SSM params stay in fp32
# Pytorch lacks support for complex32 (i.e. complex<float16>) and complex<bfloat16>.
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm)):
        module.to(dtype=dtype)

def cross_entropy_loss(yHat, y):
    if y == 1:
      return -np.log(yHat)
    else:
      return -np.log(1 - yHat)

def normalize_probs(probs):
    return probs /np.sum(probs)
# Load data
with open(args.examples, "r") as corp, open(args.labels) as targ:
    data = np.array(corp.read().split("\n"))[:-1] # Change this based on how annoying your files are
    labels = np.array(targ.read().split("\n"))[:-1]

unique_labels=np.unique(labels)
ce_mean = np.zeros(args.iters)
ce_stdev = np.zeros(args.iters)
word_maps = tokenizer.batch_decode([i for i in range(50000)])
samples = np.stack((data, labels)).T
for j in tqdm(range(args.iters)):
    # Sampling
    np.random.shuffle(samples)
    sample = samples[:args.size]
    targ_x, targ_y = sample[-1]
    label_val = np.where(unique_labels == targ_y)[0][0]
    sample=sample[:-1]
    ces=np.zeros(args.permute_count)
    for permutation in range(args.permute_count):
        np.random.shuffle(sample)
        input_x = sample[:, 0]
        input_y = sample[:, 1]

        # ICL Prompting
        prompt = '\n'.join(["%s %s %s" % (input_x[i], args.separator,input_y[i]) for i in range(len(input_x))])+"\n"
        prompt += args.query.replace("<QUERY>", targ_x)

        # Perform Inference
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device=device)
        with torch.inference_mode():
          logits = model(input_ids=input_ids).logits[:,-1].squeeze()
        logits = normalize_probs(np.array([logits[word_maps.index(label)].cpu() for label in unique_labels]))
        ces[permutation] = cross_entropy_loss(logits[1],label_val)
        del logits
    ce_mean[j] = np.mean(ces)
    ce_stdev[j] = np.std(ces)

    # Free up space
    del sample

# Post processing output
np.savetxt(args.out_path+"permute_%s_%s_%s.csv"%(args.size, args.iters, args.permute_count), np.stack([ce_mean, ce_stdev]).T, delimiter=",", header=",".join(["Mean CrossEntropy", "Stdev CrossEntropy"]))
