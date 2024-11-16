import argparse
import logging
import os
import os.path

from Converter import Converter
from run_pretrained_openfold import main
from compare import protein_to_rna, rna_to_rna

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

def train(args, epochs=50, batch_size=32, 
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
                
            ###################################
#           PROGRAM IS KNOWN TO WORK UNTIL HERE
            ###################################
            
            # LAYER 2: FOLDING
            loss = np.array([])
            for i in range(len(final_seqs)):
                out_prot = main(args, final_seqs.values()[i], save=False)
                
                if not substitute:
                    loss = protein_to_rna(out_prot, structs[i], tm_score)
                else:
                    # LAYER 3: SUBSTITUTION
                    pass # May not be necessary - I'll begin testing without the substitution layer.
                    loss = rna_to_rna(out_prot, get_structure(structs[i]), tm_score)

            loss = torch.Tensor(np.mean(loss))
            loss.backward()
            optimizer.step()
    
    torch.save(conv.state_dict(), save_path)
    
            
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_dir", type=str,
        help="Path to directory containing FASTA files, one sequence per file"
    )
    parser.add_argument(
        "template_mmcif_dir", type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments", type=str, default=None,
        help="""Path to alignment directory. If provided, alignment computation 
                is skipped and database path arguments are ignored."""
    )
    parser.add_argument(
        "--use_custom_template", action="store_true", default=False,
        help="""Use mmcif given with "template_mmcif_dir" argument as template input."""
    )
    parser.add_argument(
        "--use_single_seq_mode", action="store_true", default=False,
        help="""Use single sequence embeddings instead of MSAs."""
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--model_device", type=str, default="cpu",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--config_preset", type=str, default="model_1_ptm",
        help="""Name of a model config preset defined in openfold/config.py"""
    )
    parser.add_argument(
        "--jax_param_path", type=str, default=None,
        help="""Path to JAX model parameters. If None, and openfold_checkpoint_path
             is also None, parameters are selected automatically according to 
             the model name from openfold/resources/params"""
    )
    parser.add_argument(
        "--openfold_checkpoint_path", type=str, default=None,
        help="""Path to OpenFold checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--save_outputs", action="store_true", default=False,
        help="Whether to save all model outputs, including embeddings, etc."
    )
    parser.add_argument(
        "--cpus", type=int, default=4,
        help="""Number of CPUs with which to run alignment tools"""
    )
    parser.add_argument(
        "--preset", type=str, default='full_dbs',
        choices=('reduced_dbs', 'full_dbs')
    )
    parser.add_argument(
        "--output_postfix", type=str, default=None,
        help="""Postfix for output prediction filenames"""
    )
    parser.add_argument(
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--skip_relaxation", action="store_true", default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap", type=int, default=200,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--trace_model", action="store_true", default=False,
        help="""Whether to convert parts of each model to TorchScript.
                Significantly improves runtime at the cost of lengthy
                'compilation.' Useful for large batch jobs."""
    )
    parser.add_argument(
        "--subtract_plddt", action="store_true", default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself"""
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    parser.add_argument(
        "--cif_output", action="store_true", default=False,
        help="Output predicted models in ModelCIF format instead of PDB format (default)"
    )
    parser.add_argument(
        "--experiment_config_json", default="", help="Path to a json file with custom config values to overwrite config setting",
    )
    parser.add_argument(
        "--use_deepspeed_evoformer_attention", action="store_true", default=False, 
        help="Whether to use the DeepSpeed evoformer attention layer. Must have deepspeed installed in the environment.",
    )
    add_data_args(parser)
    args = parser.parse_args()

    if args.jax_param_path is None and args.openfold_checkpoint_path is None:
        args.jax_param_path = os.path.join(
            "openfold", "resources", "params",
            "params_" + args.config_preset + ".npz"
        )

    if args.model_device == "cpu" and torch.cuda.is_available():
        logging.warning(
            """The model is being run on CPU. Consider specifying 
            --model_device for better performance"""
        )
    
    seqs, components, macro_tags = load_data(seq_path)
    train(args=args)
    
