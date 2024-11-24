import argparse
import logging
import os
import os.path

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

from Converter import Converter
from run_pretrained_openfold import (precompute_alignments, round_up_seqlen,
                                    generate_feature_dict, list_files_with_extensions)
from protein_to_rna import protein_to_rna
from rna_to_rna import rna_to_rna

from parse_json import parse_json

import torch
import torch.nn as nn
import math
import numpy as np
import random
import time

import warnings
warnings.filterwarnings('ignore')

from openfold.config import model_config
from openfold.data import templates, feature_pipeline, data_pipeline
from openfold.data.tools import hhsearch, hmmsearch
from openfold.np import protein
from openfold.utils.script_utils import (load_models_from_command_line, parse_fasta, run_model,
                                         prep_output, relax_protein)
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.trace_utils import (
    pad_feature_dict_seq,
    trace_model_,
)

from scripts.precompute_embeddings import EmbeddingGenerator
from scripts.utils import add_data_args

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
                
            ###################################
#           PROGRAM IS KNOWN TO WORK UNTIL HERE
            ###################################
            
            # LAYER 2: FOLDING
            loss = np.array([])
            for i in range(len(final_seqs)):
                out_prot = predict(list(final_seqs.values())[i], save=False)
                
                if not substitute:
                    loss = protein_to_rna(out_prot, structs[i], tm_score)
                else:
                    # LAYER 3: SUBSTITUTION
                    pass # May not be necessary - I'll begin testing without the substitution layer.
                    #loss = rna_to_rna(out_prot, get_structure(structs[i]), tm_score)

            loss = torch.Tensor(np.mean(loss))
            loss.backward()
            optimizer.step()
    
    torch.save(conv.state_dict(), save_path)
    
def predict(seq, save):
        # Replicate main() from run_pretrained_openfold.py but without all of the nonsense
        if save:
            os.makedirs("\\folds", exist_ok=True)
        config = model_config("model_3", 
                                long_sequence_inference=False, 
                                use_deepspeed_evoformer_attention=True)
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir="",
            max_template_date="9999-21-31",
            max_hits=config.data.predict.max_templates,
            kalign_binary_path=None,
            release_dates_path=None,
            obsolete_pdbs_path=None,
        )
        data_processor = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer
        )
        random_seed = random.randrange(2**32)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed + 1)
        feature_processor = feature_pipeline.FeaturePipeline(config.data)
        alignment_dir = os.path.join("\\folds", "alignments")
        
        tag_list = []
        seq_list = []
        
        for tag, seq in zip(seqs.keys(), seqs.items()):
                tag_list.append((tag,[tag]))
                seq_list.append(seq)
        
        seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
        sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
        feature_dicts = {}
        
        model_generator = load_models_from_command_line(
            config,
            torch.device(),
            "openfold\\resources\openfold_soloseq_params\seq_model_esm1b_noptm.pt",
            None,
            os.getcwd())
        
        for model, output_directory in model_generator:
            cur_tracing_interval = 0
            for (tag, tags), seqs in sorted_targets:
                output_name = f'{tag}_{"model_1_ptm"}'

                # Does nothing if the alignments have already been computed
                precompute_alignments(tags, seqs, alignment_dir,output_directory,logger)

                feature_dict = feature_dicts.get(tag, None)
                if feature_dict is None:
                    feature_dict = generate_feature_dict( ## And this
                        tags,
                        seqs,
                        alignment_dir,
                        data_processor,
                        output_directory
                    )
                    n = feature_dict["aatype"].shape[-2]
                    rounded_seqlen = round_up_seqlen(n)
                    feature_dict = pad_feature_dict_seq(
                        feature_dict, rounded_seqlen,
                    )

                    feature_dicts[tag] = feature_dict
                processed_feature_dict = feature_processor.process_features(
                    feature_dict, mode='predict', is_multimer=False
                )

                processed_feature_dict = {
                    k: torch.as_tensor(v, device=torch.device())
                    for k, v in processed_feature_dict.items()
                }

                if rounded_seqlen > cur_tracing_interval:
                    logger.info(
                        f"Tracing model at {rounded_seqlen} residues..."
                    )
                    t = time.perf_counter()
                    trace_model_(model, processed_feature_dict)
                    tracing_time = time.perf_counter() - t
                    logger.info(
                        f"Tracing time: {tracing_time}"
                    )
                    cur_tracing_interval = rounded_seqlen

                with torch.no_grad():
                    out = model(processed_feature_dict)
                    out = tensor_tree_map(lambda t: t[..., -1], out)
                    out = relax_protein.process_outputs(
                        out, processed_feature_dict["aatype"], processed_feature_dict["seq_mask"]
                    )
                    unrelaxed_protein = relax_protein.to_pdb(out)
                    if save:
                        with open(os.path.join(output_directory, f'{output_name}_unrelaxed_model_{cur_tracing_interval}_residues.pdb'), 'w') as f:
                            f.write(unrelaxed_protein)
                    else:
                        return unrelaxed_protein
                
        
        pass
            
if __name__=="__main__":
    seqs, components, macro_tags = load_data(seq_path)
    train()
    
