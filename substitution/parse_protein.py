import sys
from Monomer import *
from gemmi import *
from openfold.np import protein
from openfold.np.protein import to_modelcif
import numpy as np

def parse_protein(path):
    try:
        doc = cif.read_file(path)
        block = doc.sole_block()
        all_xs = [element for element in block.find_loop("_atom_site.Cartn_x")]
        all_ys = [element for element in block.find_loop("_atom_site.Cartn_y")]
        all_zs = [element for element in block.find_loop("_atom_site.Cartn_z")]
        all_atoms = [element for element in block.find_loop("_atom_site.label_atom_id")]
        
        points = []
        angle_points = []
        norms = []
        
        for x, y, z, atom in zip(all_xs, all_ys, all_zs, all_atoms):
            x = float(x)
            y = float(y)
            z = float(z)

            if atom == "CA":
                points.append(np.array([x, y, z]))
                angle_points.append(np.array([x, y, z]))
            elif atom == "N":
                angle_points.append(np.array([x, y, z]))
            elif atom == "C":
                angle_points.append(np.array([x, y, z]))
                v1 = angle_points[-1]-angle_points[-2]
                v2 = angle_points[-3]-angle_points[-2]
                norms.append(np.cross(v1, v2))
                angle_points = []
        
        return points, norms
            
    except Exception as e:
        print("Oops. %s" % e)
        sys.exit(1)

def parse_protein(protein: protein.Protein):
    try:
        # Extract atom positions and atom types
        atom_positions = protein.atom_positions  # Shape: [num_res, num_atoms, 3]
        atom_mask = protein.atom_mask  # Shape: [num_res, num_atoms]
        
        points = []
        angle_points = []
        norms = []
        
        for res_idx in range(protein.num_residues):
            ca_idx = protein.atom_types['CA']
            n_idx = protein.atom_types['N']
            c_idx = protein.atom_types['C']
            
            ca_pos = atom_positions[res_idx, ca_idx]
            n_pos = atom_positions[res_idx, n_idx]
            c_pos = atom_positions[res_idx, c_idx]
            
            if atom_mask[res_idx, ca_idx]:
                points.append(ca_pos)
                angle_points.append(ca_pos)
            
            if atom_mask[res_idx, n_idx]:
                angle_points.append(n_pos)
            
            if atom_mask[res_idx, c_idx]:
                angle_points.append(c_pos)
            
            if len(angle_points) == 3:
                v1 = angle_points[2] - angle_points[1]
                v2 = angle_points[0] - angle_points[1]
                norms.append(np.cross(v1, v2))
                angle_points = []
        
        return np.array(points), np.array(norms)
            
    except Exception as e:
        print(f"Oops. {e}")
        sys.exit(1)
