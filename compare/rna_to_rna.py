from substitution import parse_protein, parse_RNA
from loss import tm_score, RMSD

def rna_to_rna(generated, template_path, tm_score=False):
    gen_points, _ = generated.data()
    temp_points, _ = parse_RNA(template_path)
    assert (len(gen_points) == len(temp_points)), "RNA structures are not the same length. Check that the input files are correct."
    if tm_score:
        return tm_score(gen_points, temp_points)
    else:
        return RMSD(gen_points, temp_points)