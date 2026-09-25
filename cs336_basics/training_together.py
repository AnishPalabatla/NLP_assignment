import torch
import numpy as np
import argparse
import wandb
import os
import json
from cs336_basics.data_loading import data_load
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.checkpointing import save_checkpoint,load_checkpoint
from cs336_basics.learning_rate_schedule import cosine_annealing
from cs336_basics.gradient_clipping import gradient_clipping


parser = argparse.ArgumentParser()

parser.add_argument("--config",type=str,required=True)
parser.add_argument("--input",type=str,required=True)
parser.add_argument("--validation",type=str,required=True)
parser.add_argument("--checkpoint",type=str,required=True)

args=parser.parse_args()

with open(args.config,"r") as f:
    config = json.load(f)

for key,value in config.items():
    setattr(args, key, value)

if args.batch_size%args.micro_batch_size != 0:
    raise ValueError(
        "batch_size must be divisible by micro_batch_size"
    )

grad_accum_steps=args.batch_size // args.micro_batch_size

print(f"Effective batch size: {args.batch_size}")
print(f"Micro batch size: {args.micro_batch_size}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

wandb.init(
    project="cs336",
    name=os.path.basename(args.config).replace(".json", "")
)

input_data=np.memmap(
    args.input,
    dtype=np.uint16,
    mode="r"
)

validation_data=np.memmap(
    args.validation,
    dtype=np.uint16,
    mode="r"
)

transformer_lm=TransformerLM(
    args.vocab_size,
    args.context_length,
    args.d_model,
    args.num_layers,
    args.num_heads,
    args.d_ff,
    args.rope_theta,
).to(device)

optimizer=AdamW(
    transformer_lm.parameters(),
    args.lr,
    args.betas,
    args.eps,
    args.weight_decay,
)

if os.path.exists(args.checkpoint):
    iteration=load_checkpoint(
        args.checkpoint,
        transformer_lm,
        optimizer,
    )
else:
    iteration=0


@torch.no_grad()
def evaluate(model,dataset,num_batches):
    model.eval()

    total_loss=0.0

    for _ in range(num_batches):
        batch_data=data_load(
            dataset,
            args.micro_batch_size,
            args.context_length,
            device,
        )

        logits=model(batch_data[0])

        loss=cross_entropy(
            logits,
            batch_data[1],
        )

        total_loss+=loss.item()

    average_loss=total_loss/num_batches

    model.train()

    return average_loss


transformer_lm.train()

for t in range(iteration,args.max_step):

    lr=cosine_annealing(
        t,
        args.lr,
        args.min_learning_rate,
        args.warmup_iters,
        args.cosine_cycle_iters,
    )

    for group in optimizer.param_groups:
        group["lr"]=lr

    optimizer.zero_grad()

    total_loss=0.0

    for _ in range(grad_accum_steps):

        batch_data=data_load(
            input_data,
            args.micro_batch_size,
            args.context_length,
            device,
        )

        logits=transformer_lm(
            batch_data[0]
        )

        micro_loss=cross_entropy(
            logits,
            batch_data[1],
        )

        total_loss+=micro_loss.item()

        micro_loss=micro_loss/grad_accum_steps

        micro_loss.backward()

    loss=total_loss/grad_accum_steps

    gradient_clipping(
        transformer_lm.parameters(),
        args.max_l2_norm,
    )

    optimizer.step()

    wandb.log({
        "train_loss": loss,
        "learning_rate": lr,
        "iteration": t,
    })

    print(
        f"iteration {t}:"
        f"train_loss={loss:.4f},"
        f"lr={lr:.6e}"
    )

    if t%args.eval_interval==0:

        validation_loss=evaluate(
            transformer_lm,
            validation_data,
            args.eval_batches,
        )

        wandb.log({
            "validation_loss": validation_loss,
            "iteration": t,
        })

        print(
            f"iteration {t}:"
            f"validation_loss={validation_loss:.4f}"
        )

    if t%100==0 or t==args.max_step-1:
        save_checkpoint(
            transformer_lm,
            optimizer,
            t+1,
            args.checkpoint,
        )
