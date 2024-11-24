import os
import os.path
import subprocess

from Converter import Converter
from run_pretrained_openfold import main
from protein_to_rna import protein_to_rna
from rna_to_rna import rna_to_rna

from parse_json import parse_json
from scripts.utils import add_data_args

import torch
import torch.nn as nn
import math
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Index to amino acid dictionary
# Largely arbitrary, but must stay consistent for any given converter
AA_DICT = {
    0: "A",
    1: "R",
    2: "N",
    3: "D",
    4: "C",
    5: "Q",
    6: "E",
    7: "G",
    8: "H",
    9: "I",
    10: "L",
    11: "K",
    12: "M",
    13: "F",
    14: "P",
    15: "S",
    16: "T",
    17: "W",
    18: "Y",
    19: "V"
}

max_len = 150

seqs = {}
components = []
macro_tags = []

seq_path = "test_RNA.json"
struct_path = ""

def load_data(path):
    seqs, components, macro_tags=parse_json(path, max_len=max_len)
    return seqs, components, macro_tags

def batch_data(iterable, n=1):
    l = len(iterable)
    iter = [(t, s) for t, s in list(iterable.items())]
    for ndx in range(0, l, n):
        yield iter[ndx:min(ndx + n, l)]

def encode_rna(seq):
    # Convert RNA sequence to nums to feed into Converter
    out = []
    for i in seq:
        if i=="A":
            out.append([0])
        elif i=="U":
            out.append([1])
        elif i=="C":
            out.append([2])
        elif i=="G":
            out.append([3])
            
    return out

def write_fastas(seqs):
    for tag, seq in list(seqs.items()):
        f = open(f"FASTAs/{tag}.fasta", "w")
        f.write(f">{tag}\n{seq}")
        f.close()

def empty_dir(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))

def get_structure(tag, path=struct_path):
    # Return the structure of an RNA molecule given its tag and the path to its structure file
    # File directory:
    # root
    #  -- openfoldRNA.dir
    #  -- train_openfoldRNA.py
    #  -- data.dir
    #  ---- component 1.dir
    #  ------ tag 1.dir
    #  -------- tag 1a.cif
    #  -------- tag 1b.cif
    # ...
    index = list(seqs.keys()).index(tag)
    component = components[index]
    macro_tag = macro_tags[index]
    
    path = f"{path}\{component}\{macro_tag}\{tag}.cif"
    return path

def train(epochs=50, batch_size=32, 
          c=None, substitute=False, tm_score=False, 
          save_path="bin\converter.pt"):
    
    # Set up converter
    if c is None:
        conv = Converter(max_seq_len=max_len)
    else:
        conv = c

    conv.train()
    optimizer = torch.optim.Adam(conv.parameters(), lr=1e-3)
    
    for _ in range(epochs):
        for batch in batch_data(seqs, batch_size):
            optimizer.zero_grad()
            # batch: ([(tag, seq), (tag, seq),...])
            
            # LAYER 1: RNA-AMINO CONVERSION
            tags = [s[0] for s in batch]
            structs = [get_structure(tags[i]) for i in range(len(tags))]
            
            # Check that structure files exist
            # if not os.path.isfile(get_structure(tags[0])):
            #     continue
            
            processed_seqs = [torch.Tensor(np.transpose(np.array(encode_rna(s[1])), (0,1))) for s in batch] # (batch, seq, base)
            aa_seqs = [conv(s) for s in processed_seqs][0] # (seq, batch, aa)
            temp = []
            
            # Reconvert to letter representation
            for i in range(len(aa_seqs)):
                temp.append(''.join([AA_DICT[n] for n in aa_seqs[i]]))
                    
            aa_seqs = temp # (seq: String, batch)

            final_seqs = {} # {tag: seq}
            for i in range(len(tags)):
                final_seqs[tags[i]] = aa_seqs[i]
                
            write_fastas(final_seqs)
            ###################################
#           PROGRAM IS KNOWN TO WORK UNTIL HERE
            ###################################
            
            # LAYER 2: FOLDING
            try:
                loss = np.array([])
                for i in range(len(final_seqs)):
                    out_prot = None#main(args, final_seqs.values()[i], save=False) # Directly calling fn may not work
                                                                            # Instead, I'll try out using cmds for it
                    
                    proc = subprocess.Popen(['python', 'run_pretrained_openfold.py', 
                                             f'--fasta_path FASTAs/{[final_seqs.keys()][i]}',
                                             '--config_preset model_3',
                                             '--use_single_seq_mode',
                                             "--template_mmcif_dir /substitution/templates"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = proc.communicate()
                    out_prot = out
                    print(out_prot)
                    if not substitute:
                        loss = protein_to_rna(out_prot, structs[i], tm_score)
                    else:
                        # LAYER 3: SUBSTITUTION
                        pass # May not be necessary - I'll begin testing without the substitution layer.
                        #loss = rna_to_rna(out_prot, get_structure(structs[i]), tm_score)

                loss = torch.Tensor(np.mean(loss))
                loss.backward()
                optimizer.step()
            except:
                empty_dir("FASTAs")
                return # Remove when done debugging/testing
    
    torch.save(conv.state_dict(), save_path)
    
            
if __name__=="__main__":
    seqs, components, macro_tags = load_data(seq_path)
    train()
    
