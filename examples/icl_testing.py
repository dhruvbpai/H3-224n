import argparse

import torch

from tqdm import trange

import numpy as np

from transformers import GPT2Tokenizer

from src.models.ssm_seq import SSMLMHeadModel


parser = argparse.ArgumentParser(description='H3 text generation')
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



with open(args.examples, "r") as corp, open(args.labels) as targ:
    data = np.array(corp.read().split("\n")) # Change this based on how annoying your files are
    labels = np.array(targ.read().split("\n"))
print(data.shape,labels.shape)
unique_labels=np.unique(labels)
preds = np.zeros(args.iters)
targs = np.zeros(args.iters)
for iter in trange(args.iters):
    sample = np.random.permutation(np.stack((data, labels)))[:args.size+1]
    targ_x, targ_y = sample[-1]
    input_x, input_y = sample[:args.size]
    prompt = '\n'.join(["%s %s %s" % (input_x[i], args.separator,input_y[i]) for i in range(len(input_x))])
    prompt += args.query.replace("<QUERY>", targ_x)
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device=device)
    out = model.forward(input_ids=input_ids)
    logits = getattr('CausalLMOutput', out)
    preds = [logits[label] for label in unique_labels]
    preds[iter] = np.where(preds == np.max(preds), preds)
    targs[iter] = np.where(unique_labels == targ_y, unique_labels)

# Post processing nonsense
np.savetxt(args.out_path+"out.txt", np.stack([preds, targs]))
