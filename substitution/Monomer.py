import numpy as np
import copy
from gemmi import *


class Monomer:
    def __init__(self, macro):
        self.atoms = dict()
        self.name = ""
        self.macro = macro
        
    def add_atom(self, x, y, z, e):
        self.atoms[e] = np.array([x, y, z])
        
    def element(self, e):
        if e[0]=="\"":
            return e[1]
        else:
            return e[0]
        
    def get_coords(self):
        return self.coords
    
    def add_name(self, name):
        self.name = name
    
    def apply_transformation(self, x, y, z):
        out = self
        for atom in out.atoms:
            out.atoms[atom] += [x,y,z]
        return out
    
    def calculate_normal(self):
        # Get the triangle vertices
        c4_pos = np.array(self.atoms['"C4\'"'])
        c1_pos = np.array(self.atoms['"C1\'"'])
        translation = -np.array(self.atoms['P'])
            
        # Calculate triangle vectors
        p_to_c4 = c4_pos + translation  # Vector from P to C4'
        p_to_c1 = c1_pos + translation  # Vector from P to C1'
        
        # Calculate normal to triangle
        normal = np.cross(p_to_c4, p_to_c1)
        normal = normal / np.linalg.norm(normal)
        return normal
    
    def align_triangle_to_xy(self):
        """
        Aligns the triangle formed by P, C1', and C4' atoms to the xy plane,
        with the P-C4' vector aligned to the x-axis.
        Returns a new aligned Monomer instance.
        """
        out = copy.deepcopy(self)
        
        try:
            # Get the triangle vertices
            c4_pos = np.array(out.atoms['"C4\'"'])
            c1_pos = np.array(out.atoms['"C1\'"'])
            translation = -np.array(out.atoms['P'])
                
            # Calculate triangle vectors
            p_to_c4 = c4_pos + translation  # Vector from P to C4'
            p_to_c1 = c1_pos + translation  # Vector from P to C1'
            
            # Calculate normal to triangle
            normal = np.cross(p_to_c4, p_to_c1)
            normal = normal / np.linalg.norm(normal)
            
            # Calculate rotation to align normal with z-axis
            z_axis = np.array([0, 0, 1])
            angle = np.arccos(np.dot(normal, z_axis))
            rotation_axis = np.cross(normal, z_axis)
            
            if np.linalg.norm(rotation_axis) > 1e-10:  # Check if vectors aren't parallel
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                
                # Create rotation matrix using Rodrigues' rotation formula
                cos_theta = np.cos(angle)
                sin_theta = np.sin(angle)
                K = np.array([
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0]
                ])
                R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
                
                # Apply first rotation
                for atom in out.atoms:
                    out.atoms[atom] = np.dot(R, out.atoms[atom])
            
            # Now align P-C4' with x-axis by rotating around z-axis
            p_to_c4_new = out.atoms['"C4\'"']  # P is at origin
            p_to_c4_xy = p_to_c4_new[:2]  # Project onto xy plane
            p_to_c4_xy = p_to_c4_xy / np.linalg.norm(p_to_c4_xy)
            
            # Calculate angle with x-axis
            x_axis = np.array([1, 0])
            angle_xy = np.arccos(np.dot(p_to_c4_xy, x_axis))
            
            # Determine rotation direction
            if p_to_c4_xy[1] < 0:
                angle_xy = -angle_xy
                
            # Create and apply z-axis rotation matrix
            cos_theta = np.cos(angle_xy)
            sin_theta = np.sin(angle_xy)
            R_z = np.array([
                [cos_theta, -sin_theta, 0],
                [sin_theta, cos_theta, 0],
                [0, 0, 1]
            ])
            
            # Apply second rotation
            for atom in out.atoms:
                out.atoms[atom] = np.dot(R_z, out.atoms[atom])
                
            return out
            
        except KeyError as e:
            raise KeyError(f"Required atom {e} not found in this monomer")

    def align_to_normal(self, target_normal):
        """
        Rotates the monomer so that the normal vector of its P-C1'-C4' triangle 
        aligns with the given target normal vector.
        
        Args:
            target_normal (np.ndarray): The target normal vector to align with (should be normalized)
            
        Returns:
            Monomer: A new Monomer instance with the rotated coordinates
        """
        out = copy.deepcopy(self)
        
        try:
            # Get current normal vector
            current_normal = self.calculate_normal()
            
            # Normalize target vector
            target_normal = target_normal / np.linalg.norm(target_normal)
            
            # Calculate rotation axis and angle
            rotation_axis = np.cross(current_normal, target_normal)
            
            # If vectors are parallel (or anti-parallel), rotation axis will be zero
            if np.linalg.norm(rotation_axis) < 1e-10:
                # If normals are anti-parallel, rotate 180° around any perpendicular axis
                if np.dot(current_normal, target_normal) < 0:
                    # Find a perpendicular vector to rotate around
                    if abs(current_normal[0]) < abs(current_normal[1]):
                        rotation_axis = np.cross(current_normal, [1, 0, 0])
                    else:
                        rotation_axis = np.cross(current_normal, [0, 1, 0])
                    angle = np.pi
                else:
                    # Vectors are already aligned
                    return out
            else:
                # Calculate rotation angle
                angle = np.arccos(np.clip(np.dot(current_normal, target_normal), -1.0, 1.0))
            
            # Normalize rotation axis
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            # Create rotation matrix using Rodrigues' rotation formula
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            R = np.eye(3) + sin_theta * K + (1 - cos_theta) * np.dot(K, K)
            
            # Apply rotation to all atoms
            for atom in out.atoms:
                out.atoms[atom] = np.dot(R, out.atoms[atom])
                
            return out
            
        except KeyError as e:
            raise KeyError(f"Required atom {e} not found in this monomer")

    
    def load_template(self, n):
        if n=="A": path = "templates/Adenine_template.cif"
        elif n=="C": path = "templates/Cytosine_template.cif"
        elif n=="G": path = "templates/Guanine_template.cif"
        elif n=="U": path = "templates/Uracil_template.cif"
        atoms = []
        atom_xs = []
        atom_ys = []
        atom_zs = []
        try:
            doc = cif.read_file(path)  # copy all the data from mmCIF file
            block = doc.sole_block()  # mmCIF has exactly one block
            for element in block.find_loop("_atom_site.label_atom_id"):
                atoms.append(element)
            for element in block.find_loop("_atom_site.Cartn_x"):
                atom_xs.append(float(element))
            for element in block.find_loop("_atom_site.Cartn_y"):
                atom_ys.append(float(element))
            for element in block.find_loop("_atom_site.Cartn_z"):
                atom_zs.append(float(element))
        except Exception as e:
            print("Oops. %s" % e)
        for i in range(len(atoms)):
            self.add_atom(atom_xs[i], atom_ys[i], atom_zs[i], atoms[i])
        
        
    def __str__(self, start=1):
        #Return what this monomer would look like in an mmCIF file#
        out = ""
        c = start
        for i in self.atoms:
            out += f"ATOM {c}\t{self.element(i)}\t{i}\t{self.name}\t. A 1 1\t?\t{round(self.atoms[i][0],3)}\t{round(self.atoms[i][1],3)}\t{round(self.atoms[i][2],3)}\n"
            c += 1
        out += "\b\b"
        return out
    
    def print(self, start=1, number=1):
        #Return what this monomer would look like in an mmCIF file#
        out = ""
        c = start
        for i in self.atoms:
            out += f"ATOM {c}\t{self.element(i)}\t{i}\t{self.name}\t. A 1 {number}\t?\t{round(self.atoms[i][0],3)}\t{round(self.atoms[i][1],3)}\t{round(self.atoms[i][2],3)}\n"
            c += 1
        return out
    
    def __len__(self):
        return len(self.atoms)