from loss import RMSD, tm_score
from parse_RNAs import parse_rna
from parse_proteins import parse_protein

def protein_to_rna(protein, rna_path, tm=False):
    prot_points, _ = parse_protein(protein)
    rna_points, _ = parse_rna(rna_path)
    if tm:
        return tm_score(prot_points, rna_points)
    else:
        return RMSD(prot_points, rna_points)