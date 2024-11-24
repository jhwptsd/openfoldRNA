## For precomputing embedding alignments
import os

def make_dir(seqs):
    os.makedirs("FASTAs", exist_ok=True)
    for tag, seq in list(seqs.items()):
        with open(f"FASTAs/{tag}.fasta", "w") as f:
            f.write(f">{tag}\n{seq}")
            
