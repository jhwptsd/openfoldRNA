from parse_RNAs import parse_RNA
from loss import RMSD, tm_score

def rna_to_rna(generated, template_path, tm=False):
    gen_points, _ = generated.data()
    temp_points, _ = parse_RNA(template_path)
    if tm:
        return tm_score(gen_points, temp_points)
    else:
        return RMSD(gen_points, temp_points)