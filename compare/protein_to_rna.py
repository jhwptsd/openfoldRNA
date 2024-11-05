from substitution import parse_protein, parse_RNA
from loss import tm_score, RMSD

def protein_to_rna(protein, rna_path, tm_score=False):
    prot_points, _ = parse_protein(protein)
    rna_points, _ = parse_RNA(rna_path)
    assert (len(prot_points) == len(rna_points)), "Protein and RNA structures are not the same length. Check that the input files are correct."
    if tm_score:
        return tm_score(prot_points, rna_points)
    else:
        return RMSD(prot_points, rna_points)