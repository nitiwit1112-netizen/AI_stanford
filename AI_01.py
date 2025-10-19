from collections import defaultdict
from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from jaxtyping import Float, Int
import requests
# --------------------------------------Byte-Pair Encoding--------------------------------------------------------------
@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index

class ByteTokenizer:
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices
    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
        #Return `indices`, but with all instances of `pair` replaced with `new_index`.
        new_indices = []  # @inspect new_indices
        i = 0  # @inspect i
        while i < len(indices):
            if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
                new_indices.append(new_index)
                i += 2
            else:
                new_indices.append(indices[i])
            i += 1
        return new_indices

def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merge
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
        vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
        for i in range(num_merges):
            #Count the number of occurrences of each pair of tokens
            counts = defaultdict(int)
            for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
                counts[(index1, index2)] += 1  # @inspect counts

            if not counts: # prevent it from crashing out
                break

            #Find the most common pair.
            pair = max(counts, key=counts.get)  # @inspect pair
            index1, index2 = pair
            #Merge that pair.
            new_index = 256 + i  # @inspect new_index
            merges[pair] = new_index  # @inspect merges
            vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
            indices = merge(indices, pair, new_index)  # @inspect indices
        return BPETokenizerParams(vocab=vocab, merges=merges)

# ------------------ MODEL FRAMEWORK -----------------------------------------------------------------------------------
############################################################

def get_device(index):
    #"Use GPU if available."
    if torch.cuda.is_available():
        return torch.device(f"cuda:{index}")
    else:
        return torch.device("cpu")

def get_num_parameters(model):
    """Count model parameters."""
    return sum(param.numel() for param in model.parameters())

def note_about_randomness(seed):
    """Set all random seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class Linear(nn.Module):
    """Simple linear layer using nn.Parameter."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim) / np.sqrt(input_dim))

    def forward(self, x):
        return x @ self.weight


class Cruncher(nn.Module):
    """Deep linear model."""
    def __init__(self, dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([Linear(dim, dim) for _ in range(num_layers)])
        self.final = Linear(dim, 1)

    def forward(self, x):
        B, D = x.size()
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        x = x.squeeze(-1)
        return x

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                if wd != 0:
                    p.data.add_(p.data, alpha=-lr * wd)

                p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

class AdaGrad(torch.optim.Optimizer):
    #"Custom AdaGrad Optimizer."
    def __init__(self, params, lr):
        super(AdaGrad, self).__init__(params, dict(lr=lr))

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                g2 = state.get("g2", torch.zeros_like(grad))
                g2 += torch.square(grad)
                state["g2"] = g2
                p.data -= lr * grad / torch.sqrt(g2 + 1e-5)

def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of the jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Step 1: compute log probabilities from logits (stable version)
    log_probs = F.log_softmax(inputs, dim=-1)  # shape: (batch_size, vocab_size)

    # Step 2: select log prob for the correct class for each example
    target_log_probs = log_probs[torch.arange(inputs.size(0)), targets]

    # Step 3: compute mean negative log likelihood
    loss = -target_log_probs.mean()

    return loss

# ------------------ DATA PROCESSING ----------------------------------------------------------------------------------

def get_batch(data, batch_size, sequence_length, device):
    #"Randomly sample contiguous subsequences."
    start_indices = torch.randint(len(data) - sequence_length, (batch_size,))
    x = torch.tensor([data[start:start + sequence_length] for start in start_indices])
    if torch.cuda.is_available():
        x = x.pin_memory()
    x = x.to(device, non_blocking=True)
    return x

def data_loading(tokens):
    #"Save BPE tokenized text as mammap data file."
    np.save("data.npy", np.array(tokens, dtype=np.int32))
    data = np.memmap("data.npy", dtype=np.int32)
    return data

# ------------------- TRAINING LOOP ------------------------------------------------------------------------------------

def train_model(data, num_layers, D, lr, steps):
    #"Train Cruncher model using AdamW."
    device = get_device(0)
    model = Cruncher(dim=D, num_layers=num_layers).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    print(f"Training on device: {device}")
    print(f"Total parameters: {get_num_parameters(model)}")

    losses = []

    for step in range(steps):
        x = get_batch(data, 8, D, device)
        y = torch.roll(x, -1, dims=1)[:, 0].float()
        x = x.float()
        pred = model(x)
        loss = F.cross_entropy(pred, y)
        op   timizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if step % 20 == 0:
            print(f"Step {step:03d} | Loss: {loss.item():.15f}")

    print("Training complete.")
    # Plot loss curve
    plt.plot(losses, label="Training Loss")
    plt.title("Training Loss Over Time")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
    return model, optimizer

# ------------------- CHECKPOINTING ------------------------------------------------------------------------------------

def checkpointing(model, optimizer):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, "model_checkpoint.pt")
    print("Checkpoint saved.")
    loaded = torch.load("model_checkpoint.pt")
    print("Checkpoint loaded.")
    return loaded

# -------------------- MAIN --------------------------------------------------------------------------------------------

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

def main():
    # ---------------- SETUP ----------------
    import os
    import requests

    seed = 42
    note_about_randomness(seed)

    # download tiny Shakespeare dataset if not present
    if not os.path.exists("shakespeare.txt"):
        print("Downloading Shakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        text = requests.get(url).text
        with open("shakespeare.txt", "w", encoding="utf-8") as f:
            f.write(text)
    else:
        with open("shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()

    # use only a small portion to fit in 8 GB RAM
    text = text[:100000]  # first 100k characters (~10%)

    print(f"Dataset length: {len(text)} characters")

    # ---------------- TOKENIZER ----------------
    num_merges = 100
    params = train_bpe(text, num_merges)
    btok = ByteTokenizer()
    indices = btok.encode(text)
    for pair, new_idx in params.merges.items():
        indices = merge(indices, pair, new_idx)

    np.save("shakespeare_tokens.npy", np.array(indices, dtype=np.int32))
    data = np.memmap("shakespeare_tokens.npy", dtype=np.int32, mode="r")

    # ---------------- MODEL SETTINGS ----------------
    num_layers = 1
    D = 32
    lr = 3e-4
    steps = 300

    print(f"Training Cruncher: layers={num_layers}, D={D}, lr={lr}, steps={steps}")

    # ---------------- TRAIN ----------------
    model, optimizer = train_model(
        data=data,
        num_layers=num_layers,
        D=D,
        lr=lr,
        steps=steps
    )

    # ---------------- CHECKPOINT ----------------
    checkpointing(model, optimizer)
    # generating text
    print("✅ Training complete and model saved.")

# ------------------ ENTRY POINT ---------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()