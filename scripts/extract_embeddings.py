from pathlib import Path
import sys
import os
import time

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# --- Paths / Imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
GENOMIC_FM_DIR = REPO_ROOT / "external" / "genomic-FM"
DATA_DIR = GENOMIC_FM_DIR / "root" / "data"  # where verified_real_clinvar.csv likely lives

print("\n[1/6] Setting up import paths...")
print(f"  - REPO_ROOT:     {REPO_ROOT}")
print(f"  - GENOMIC_FM_DIR:{GENOMIC_FM_DIR}")
print(f"  - DATA_DIR:      {DATA_DIR}")

sys.path.insert(0, str(GENOMIC_FM_DIR))
os.chdir(GENOMIC_FM_DIR)


from src.dataloader.data_wrapper import RealClinVar

# --- Load data ---
Seq_length = 1024   # length of sequences to extract from ClinVar records

print("\n[2/6] Loading ClinVar data via RealClinVar...")
loader = RealClinVar(num_records=151, all_records=False)    # change number of records based on needs
data = loader.get_data(Seq_length=Seq_length)

print("  - Loaded successfully.")
print(f"  - data type: {type(data)}")
print(f"  - number of records: {len(data)}")

# --- Retrieve Sequences and Print Information ---
print("\n[3/6] Extracting Reference/Alternative sequences and labels...")
lc_reference_sequences = [record[0][0] for record in data]
lc_alternative_sequences = [record[0][1] for record in data]
lc_labels = [record[1] for record in data]

unique_labels = sorted(set(lc_labels))
print(f"  - unique labels: {unique_labels}")
print(f"  - number of reference sequences: {len(lc_reference_sequences)}")
print(f"  - number of alternative sequences: {len(lc_alternative_sequences)}")

# --- Import Tokenizer and Model ---
print("\n[4/6] Loading tokenizer + model...")
MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

print(f"  - model: {MODEL_NAME}")
print(f"  - device: {device}")
print(f"  - tokenizer.model_max_length: {tokenizer.model_max_length}")

# tokenize
print("\n[5/6] Tokenizing sequences...")
t0 = time.time()

all_sequences = lc_reference_sequences + lc_alternative_sequences
n = len(lc_reference_sequences)

enc = tokenizer(
    all_sequences,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=Seq_length,
)

input_ids = enc["input_ids"]           # (2N, L)
attention_mask = enc["attention_mask"]   # (2N, L)

print(f"  - total sequences tokenized (REF+ALT): {len(all_sequences)} = {2*n}")
print(f"  - input_ids shape: {tuple(input_ids.shape)}")
print(f"  - attention_mask shape: {tuple(attention_mask.shape)}")
print(f"  - tokenization done in {time.time() - t0:.2f}s")


# Inference and Pooling
print("\n[6/6] Running model inference and computing pooled embeddings...")
t0 = time.time()

batch_size = 2
all_embeds = []

model.eval()

with torch.no_grad():
    for i in range(0, input_ids.size(0), batch_size):
        batch_input_ids = input_ids[i:i+batch_size].to(device)
        batch_mask = attention_mask[i:i+batch_size].to(device)

        # Optional but recommended: mixed precision on GPU
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_mask,
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
            )

        last_h = outputs.hidden_states[-1]                 # (B, L, D)
        mask = batch_mask.unsqueeze(-1).to(last_h.dtype)   # (B, L, 1)
        pooled = (last_h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, D)

        all_embeds.append(pooled.float().cpu())  # keep saved embeddings in float32 for stability

        if (i // batch_size) % 10 == 0:
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            print(f"  - batch {i//batch_size:04d} | processed {i+len(batch_input_ids)}/{input_ids.size(0)} "
                  f"| GPU allocated {allocated:.1f} GiB, reserved {reserved:.1f} GiB")
            
seq_embeddings = torch.cat(all_embeds, dim=0)  # (2N, D)      

n = len(lc_reference_sequences)
ref_seq_embeddings = seq_embeddings[:n]                   # (N, D)
alt_seq_embeddings = seq_embeddings[n:]                   # (N, D)
delta_seq_embeddings = alt_seq_embeddings - ref_seq_embeddings

print(f"  - pooled seq embeddings shape (REF): {tuple(ref_seq_embeddings.shape)}")
print(f"  - pooled seq embeddings shape (ALT): {tuple(alt_seq_embeddings.shape)}")
print(f"  - pooled seq embeddings shape (ALT-REF): {tuple(delta_seq_embeddings.shape)}")
print(f"  - inference+pooling done in {time.time() - t0:.2f}s")

# --- Save ---
print("\nSaving embeddings to disk...")
DATA_DIR.mkdir(parents=True, exist_ok=True)

save_path = DATA_DIR / f"clinvar_embeddings__n{n}__len{Seq_length}__layer-1__maskedmean.pt"

payload = {
    "model_name": MODEL_NAME,
    "seq_len": Seq_length,
    "pooling": "masked_mean",
    "layer": -1,
    "labels": lc_labels,
    "ref_embeddings": ref_seq_embeddings.cpu(),
    "alt_embeddings": alt_seq_embeddings.cpu(),
    "delta_embeddings": delta_seq_embeddings.cpu(),
    # optional (comment out if you want smaller files)
    "ref_sequences": lc_reference_sequences,
    "alt_sequences": lc_alternative_sequences,
}

torch.save(payload, save_path)

print(f"  - saved: {save_path}")
print("Done.\n")