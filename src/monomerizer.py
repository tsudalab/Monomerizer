#!/usr/bin/env python3

# This script takes a isomeric SMILES file as input and outputs a seq (like fasta) file with the corresponding amino acid sequence.
# The script also outputs a isomeric SMILES file with the NNAA (non-natural amino acid) labeled as "X".
# Any compound connected to a valid backbone is considered as individual amino acid.
# The NNAAs that do not possess a valid backbone "[NH,NH2]CC(=O)O" required to continuously form peptide bonds, are considered as terminal modifications, and are named as "X0ter", "X1ter", etc.

import os
from rdkit import Chem
from rdkit.Chem import RegistrationHash
from rdkit.Chem.RegistrationHash import HashLayer
from collections import deque
import argparse
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from draw import MoleculeDrawer
from collections import defaultdict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process SMILES files and generate amino acid sequences.")
    parser.add_argument("--input_file", default="demo/example_smiles.txt", help="Input SMILES file")
    parser.add_argument("-process_cyclic", action="store_true", help="Process cyclic peptides")
    parser.add_argument("--min_amino_acids", type=int, default=3, help="Minimum number of amino acids")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size")
    parser.add_argument("--output_dir", default="output/tmp", help="Output directory")
    parser.add_argument("--max_workers", type=int, default=mp.cpu_count(), help="Maximum number of workers for parallel processing")
    parser.add_argument("-draw", action="store_true", help="Draw molecules")
    return parser.parse_args()

name_smi_dict = {
    # isomeric SMILES from pubchem. eg https://pubchem.ncbi.nlm.nih.gov/compound/Alanine except for Asp (from https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=3309) and Arg (from https://en.wikipedia.org/wiki/Arginine)
    "TrpTer": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
    "Trp": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O))N",
    "ArgTer": "C(C[C@@H](C(=O)O)N)CNC(=N)N",
    "Arg Ter": "NC(N)=NCCC[C@H](N)C(=O)O",
    "Arg": "C(C[C@@H](C(=O))N)CNC(=N)N",
    "Arg2": "NC(N)=NCCC[C@H](N)C(=O)",
    "HisTer": "C1=C(NC=N1)C[C@@H](C(=O)O)N",
    "His": "C1=C(NC=N1)C[C@@H](C(=O))N",
    "ProTer": "C1C[C@H](NC1)C(=O)O",
    "Pro": "C1C[C@H](NC1)C(=O)",
    "LysTer": "C(CCN)C[C@@H](C(=O)O)N",
    "Lys": "C(CCN)C[C@@H](C(=O))N",
    "MetTer": "CSCC[C@@H](C(=O)O)N",
    "Met": "CSCC[C@@H](C(=O))N",
    "GlnTer": "C(CC(=O)N)[C@@H](C(=O)O)N",
    "Gln": "C(CC(=O)N)[C@@H](C(=O))N",
    "AsnTer": "C([C@@H](C(=O)O)N)C(=O)N",
    "Asn": "C([C@@H](C(=O))N)C(=O)N",
    "GluTer": "C(CC(=O)O)[C@@H](C(=O)O)N",
    "Glu": "C(CC(=O)O)[C@@H](C(=O))N",
    "AspTer": "OC(=O)C[C@@H](C(=O)O)N",
    "Asp": "OC(=O)C[C@@H](C(=O))N",
    "TyrTer": "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",
    "Tyr": "C1=CC(=CC=C1C[C@@H](C(=O))N)O",
    "PheTer": "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",
    "Phe": "C1=CC=C(C=C1)C[C@@H](C(=O))N",
    "IleTer": "CC[C@H](C)[C@@H](C(=O)O)N",  # TODO add correct hydroxyl oxygen for every AA terminal
    "Ile": "CC[C@H](C)[C@@H](C(=O))N",
    "LeuTer": "CC(C)C[C@@H](C(=O)O)N",
    "Leu": "CC(C)C[C@@H](C(=O))N",
    "ValTer": "CC(C)[C@@H](C(=O)O)N",
    "Val": "CC(C)[C@@H](C(=O))N",
    "ThrTer": "C[C@H]([C@@H](C(=O)O)N)O",
    "Thr": "C[C@H]([C@@H](C(=O))N)O",
    "CysTer": "C([C@@H](C(=O)O)N)S",
    "Cys": "C([C@@H](C(=O))N)S",
    "SerTer": "C([C@@H](C(=O)O)N)O",
    "Ser": "C([C@@H](C(=O))N)O",
    "AlaTer": "C[C@@H](C(=O)O)N",
    # FBR: I wonder if we should have a SMILES for AlaStart
    "Ala": "C[C@@H](C(=O))N",
    # Saturated the carbon
    "GlyTer": "C(C(=O)O)N",
    "Gly": "C(C(=O))N",
}

smi2mol = {}
for aa_name, aa_smi in name_smi_dict.items():
    smi2mol[aa_name] = Chem.MolFromSmiles(aa_smi)

peptide_bond_mol = Chem.MolFromSmarts("[N,n][C,c]C(=O)[*!O]") # [*!O] ensures it does not match AAter
edge_C_position = 2
edge_N_position = 4
valid_backbone = Chem.MolFromSmarts("[NH,NH2]CC(=O)[OH]")
loose_backbone = Chem.MolFromSmarts("[C,c](C(=O)O)[N,n]") # Also detects backbone that contains a benzene ring. Used for removing -OH
OH_position = 3
oxygen = Chem.Atom(8)

three2one_letter = {
    "Ala": "A",
    "Gly": "G",
    "Ile": "I",
    "Leu": "L",
    "Pro": "P",
    "Val": "V",
    "Phe": "F",
    "Trp": "W",
    "Tyr": "Y",
    "Asp": "D",
    "Glu": "E",
    "Arg": "R",
    "His": "H",
    "Lys": "K",
    "Ser": "S",
    "Thr": "T",
    "Cys": "C",
    "Met": "M",
    "Asn": "N",
    "Gln": "Q",
}

aa2color_dict = {
    "Asp": (0.902, 0.039, 0.039),
    "Glu": (0.961, 0.1, 0.537),
    "Arg": (0.078, 0.353, 1),
    "Lys": (0.42, 0.353, 1),
    "His": (0.51, 0.51, 0.824),
    "Tyr": (0.196, 0.196, 0.667),
    "Phe": (0.341, 0.196, 0.667),
    "Trp": (0.706, 0.353, 0.706),
    "Asn": (0, 0.863, 0.863),
    "Gln": (0.5, 0.82, 0.863),
    "Met": (0.902, 0.902, 0),
    "Cys": (0.722, 0.902, 0),
    "Ser": (0.98, 0.588, 0),
    "Thr": (0, 0.612, 0.412),
    "Gly": (0.98, 0.922, 0.922),
    "Ala": (0.784, 0.784, 0.639),
    "Val": (0.059, 0.51, 0.059),
    "Leu": (0.29, 0.51, 0.059),
    "Ile": (0.29, 0.51, 0.471),
    "Pro": (1, 0.588, 0.51),
}

# no integer in the tuple was already matched
def tuple_fully_unmatched(indexes_group, already_matched, mol_a):
    res = True
    for i in indexes_group:
        if mol_a.GetAtomWithIdx(i).HasProp("AA") and mol_a.GetAtomWithIdx(i).GetProp(
            "AA"
        ).startswith("Unk"):
            res = False
            break
        if i in already_matched:
            res = False
            break
    return res


def match_AA(mol_b, dict):
    atoms_already_matched = set()
    for aa_name, aa_mol in dict.items():
        i = 0
        for atom_indexes_group in mol_b.GetSubstructMatches(aa_mol, useChirality=True):
            prop = aa_name + ":" + str(i)
            if tuple_fully_unmatched(atom_indexes_group, atoms_already_matched, mol_b):
                for a_i in atom_indexes_group:
                    mol_b.GetAtomWithIdx(a_i).SetProp("AA", prop)
                    atoms_already_matched.add(a_i)
                i += 1


def find_peptide_bonds(mol_c):
    atom_indices_surrounding_peptide_bond = []
    for bonded_AA in mol_c.GetSubstructMatches(peptide_bond_mol):
        C_idx = mol_c.GetAtomWithIdx(bonded_AA[edge_C_position]).GetIdx()
        N_idx = mol_c.GetAtomWithIdx(bonded_AA[edge_N_position]).GetIdx()
        atom_indices_surrounding_peptide_bond.append([C_idx, N_idx])
    return atom_indices_surrounding_peptide_bond


def set_peptide_bond_prop(mol, atom_indices_surrounding_peptide_bond):
    peptide_bonds = []
    for C_idx, N_idx in atom_indices_surrounding_peptide_bond:
        mol.GetAtomWithIdx(C_idx).SetProp("bond_site", "C")
        mol.GetAtomWithIdx(N_idx).SetProp("bond_site", "N")
        peptide_bond = mol.GetBondBetweenAtoms(C_idx, N_idx)
        peptide_bond.SetProp("bondNote", "8<")
        peptide_bond.SetProp("peptide_bond", "peptide_bond")
        peptide_bonds.append(peptide_bond.GetIdx())
    return peptide_bonds


def label_peptide_bonds(mol_e):
    atom_indices_surrounding_peptide_bond = find_peptide_bonds(mol_e)
    peptide_bonds = set_peptide_bond_prop(mol_e, atom_indices_surrounding_peptide_bond)
    return peptide_bonds


def label_NNAAs(mol_e, peptide_bonds):
    NNAA_idx = 0
    for a_i in range(mol_e.GetNumHeavyAtoms()):
        the_atom = mol_e.GetAtomWithIdx(a_i)
        if not the_atom.HasProp("AA"):
            atom_index_of_the_NNAA = the_atom.GetIdx()
            label_unmatched_NNAA(
                mol_e, atom_index_of_the_NNAA, NNAA_idx, peptide_bonds
            )
            NNAA_idx += 1
    return NNAA_idx

def prepare_graph(first_atom_index):
    queue = deque([first_atom_index])
    visited = set([first_atom_index])
    return queue, visited


def enqueue_neighbor_indices(mol_f, atom, queue, visited):
    neighbor_indices = [neighbor[1] for neighbor in get_neighbors(mol_f, atom)]
    for neighbor_atom_idx in neighbor_indices:
        if neighbor_atom_idx not in visited:
            queue.append(neighbor_atom_idx)
            visited.add(neighbor_atom_idx)
    return queue, visited


def get_neighbors(mol_g, atom):
    neighbors_and_indices = []
    for neighbor_atom in atom.GetNeighbors():
        neighbor_atom_idx = neighbor_atom.GetIdx()
        neighbor_atom = mol_g.GetAtomWithIdx(neighbor_atom_idx)
        neighbors_and_indices.append([neighbor_atom, neighbor_atom_idx])
    return neighbors_and_indices


def cross_peptide_bond(mol_f, current_atom_idx, neighbor_idx, peptide_bonds):
    bond_i = mol_f.GetBondBetweenAtoms(current_atom_idx, neighbor_idx).GetIdx()
    return bond_i in peptide_bonds


def NNAA_continues(neighbor_atom, first_AA_observed):
    return (
        neighbor_atom.HasProp("AA") == False
        or neighbor_atom.GetProp("AA") == first_AA_observed
    )


def get_current_atom_with_prop(mol_h, atom_idx_queue, prop):
    current_atom_idx = atom_idx_queue.popleft()
    current_atom = mol_h.GetAtomWithIdx(current_atom_idx)
    current_atom.SetProp("AA", prop)
    return current_atom, current_atom_idx


def label_unmatched_NNAA(mol, atom_index_of_the_NNAA, NNAA_idx, peptide_bonds):
    atom_idx_queue, visited_atoms = prepare_graph(atom_index_of_the_NNAA)
    first_AA_observed = None
    prop = f"Unk{NNAA_idx}"
    while atom_idx_queue:
        current_atom, current_atom_idx = get_current_atom_with_prop(
            mol, atom_idx_queue, prop
        )
        neighbors_and_indices = get_neighbors(mol, current_atom)
        for neighbor in neighbors_and_indices:
            neighbor_atom, neighbor_idx = neighbor
            if neighbor_idx not in visited_atoms and not cross_peptide_bond(
                mol, current_atom_idx, neighbor_idx, peptide_bonds
            ):
                visited_atoms.add(neighbor_idx)
                if NNAA_continues(neighbor_atom, first_AA_observed):
                    atom_idx_queue.append(neighbor_idx)
                elif first_AA_observed is None:  # first_AA_observed unseen
                    first_AA_observed = neighbor_atom.GetProp("AA")
                    atom_idx_queue.append(neighbor_idx)


def get_first_base_aa(mol_j, first_atom_index):
    first_atom = mol_j.GetAtomWithIdx(first_atom_index)
    current_base_aa = first_atom.GetProp("AA")
    return current_base_aa

def label_boundary_bonds(mol):
    for bond in mol.GetBonds():
        atom1_i, atom2_i, prop1, prop2 = get_connected_atoms_and_props(bond, mol)
        if (
            prop1 != prop2
        ):
            bond.SetProp("boundary", "boundary")
            mol.GetAtomWithIdx(atom1_i).SetProp("bond_site", "bond_site")
            mol.GetAtomWithIdx(atom2_i).SetProp("bond_site", "bond_site")

def add_order_to_atomNote(mol_v, aa_order, current_base_aa):
    for atom_idx in range(mol_v.GetNumAtoms()):
        atom = mol_v.GetAtomWithIdx(atom_idx)
        if atom.GetProp("AA") == current_base_aa:
            atom.SetProp("atomNote", f"{current_base_aa}:{aa_order}")

def reorder_AAs(mol_k, first_atom_index):
    atom_idx_queue, visited_atom_indices = prepare_graph(first_atom_index)
    aa_list = []
    aa_order = 1
    current_base_aa = get_first_base_aa(mol_k, first_atom_index)

    while atom_idx_queue:
        add_order_to_atomNote(mol_k, aa_order, current_base_aa)
        atom_index = atom_idx_queue.popleft()
        the_atom = mol_k.GetAtomWithIdx(atom_index)
        aa_in_question = the_atom.GetProp("AA")
        if current_base_aa != aa_in_question:
            current_base_aa, atom_idx_queue = switch_base_and_empty_queue(
                aa_list, current_base_aa, aa_in_question, atom_idx_queue, atom_index
            )
            aa_order += 1
        enqueue_neighbor_indices(mol_k, the_atom, atom_idx_queue, visited_atom_indices)

    aa_list.append(current_base_aa)  # append the last AA
    return aa_list


def switch_base_and_empty_queue(
    aa_list, current_base_aa, aa_in_question, atom_idx_queue, idx
):
    aa_list.append(current_base_aa)
    current_base_aa = aa_in_question
    atom_idx_queue = deque([idx])
    return current_base_aa, atom_idx_queue

def label_belongs_to_AA(label):
    shorter_label = label[:3]
    return shorter_label != "Unk" and not label.startswith("X")


def record_if_terminal(peptide_bonded_props, peptide_bonded_atoms, prop, atom):
    if (
        prop in peptide_bonded_props
    ):  # the peptide bond was seen twice i.e. it has both ends
        peptide_bonded_props.remove(prop)
    else:
        peptide_bonded_props.append(prop)
        peptide_bonded_atoms.append(atom)


def get_first_atom_index(mol_l, peptide_bonded_props, peptide_bonded_atoms):
    first_atom_index = 0
    for a_i in range(mol_l.GetNumAtoms()):
        a = mol_l.GetAtomWithIdx(a_i)
        if (
            a_i in peptide_bonded_atoms
            and a.GetProp("AA") in peptide_bonded_props
            and a.GetSymbol() == "C"
        ):
            first_atom_index = a_i
            break
    return first_atom_index


def mol_is_cyclic_peptide(mol_u, ignore_cyclic_peptide):
    if ignore_cyclic_peptide == False:
        return False
    for bond in mol_u.GetBonds():  # for any bond including peptide bonds
        if bond.IsInRing() and (bond.HasProp("boundary") or bond.HasProp("peptide_bond")):
            return True


def search_terminal_AA(mol_m):  # for highlight and searching terminal AA
    peptide_bonded_props, peptide_bonded_atoms = [], []
    for bond in mol_m.GetBonds():  # for any bond including peptide bonds
        atom1_i, atom2_i, prop1, prop2 = get_connected_atoms_and_props(bond, mol_m)
        if bond.HasProp(
            "peptide_bond"
        ):  # will remain in the list only if it is connected to a terminal AA
            record_if_terminal(
                peptide_bonded_props, peptide_bonded_atoms, prop1, atom1_i
            )
            record_if_terminal(
                peptide_bonded_props, peptide_bonded_atoms, prop2, atom2_i
            )
    return peptide_bonded_props, peptide_bonded_atoms


def get_connected_atoms_and_props(bond, mol_t):
    atom1_i, atom2_i = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    prop1, prop2 = mol_t.GetAtomWithIdx(atom1_i).GetProp("AA"), mol_t.GetAtomWithIdx(
        atom2_i
    ).GetProp("AA")
    return atom1_i, atom2_i, prop1, prop2

def write_seq(aa_list):
    split_seq = []
    for aa in aa_list:
        if aa[:3] == ("Unk"):
            acid = "?"
        elif aa.startswith("X"):
            acid = aa.split(":")[0]
        else:
            acid = three2one_letter[aa[:3]]
        split_seq.append(acid)
    return split_seq

def get_NNAAs(mol):
    rwmol = Chem.RWMol(mol)
    remove_peptide_bonds(rwmol) # this needs to come before remove_atoms
    remove_atoms(rwmol, mol, label_belongs_to_AA)
    try:
        return Chem.GetMolFrags(rwmol, asMols=True, sanitizeFrags=True)
    except ValueError:
        return "error"


def remove_atoms(rwmol, mol, func, **kwargs):
    atom_number = mol.GetNumAtoms() - 1
    while atom_number >= 0:
            prop = rwmol.GetAtomWithIdx(atom_number).GetProp("AA")
            if func(prop, **kwargs):
                rwmol.RemoveAtom(atom_number)
            atom_number -= 1

def add_OH(rwmol, begin_atom_idx, end_atom_idx):
    rwmol.AddAtom(oxygen)
    oxygen_idx = rwmol.GetNumAtoms() -1
    if rwmol.GetAtomWithIdx(begin_atom_idx).GetAtomicNum() == 6: # Carbon
        rwmol.AddBond(begin_atom_idx, oxygen_idx, Chem.BondType.SINGLE)
    elif rwmol.GetAtomWithIdx(end_atom_idx).GetAtomicNum() == 6: # Carbon
        rwmol.AddBond(oxygen_idx, end_atom_idx, Chem.BondType.SINGLE)


def remove_peptide_bonds(rwmol):
    current_bond_idx = rwmol.GetNumBonds() - 1
    while current_bond_idx >= 0:
        current_bond = rwmol.GetBondWithIdx(current_bond_idx)
        if current_bond.HasProp("peptide_bond") and current_bond.IsInRing() == False:
            begin_atom_idx, end_atom_idx = current_bond.GetBeginAtomIdx(), current_bond.GetEndAtomIdx()
            rwmol.RemoveBond(
                begin_atom_idx, end_atom_idx
            )
            add_OH(rwmol, begin_atom_idx, end_atom_idx)
        current_bond_idx -= 1

def detect_terminal(NNAA):
    if NNAA.HasSubstructMatch(valid_backbone):
        return "NotTer"
    else:
        return "ter" # don't use capital letter, for tokenization
    

def enlist_NNAA(new_NNAA, df, ter_or_not, bond_atom_indices):
    new_smi = Chem.MolToSmiles(new_NNAA, isomericSmiles=True, canonical=True)
    new_smi_rootedAtAtom0 = Chem.MolToSmiles(new_NNAA, isomericSmiles=True, canonical=True, rootedAtAtom=0)
    bond_atom_indices = [new_smi_rootedAtAtom0] + bond_atom_indices
    new_data = pd.DataFrame({
        'SMILES': [new_smi],
        'TERMINAL': [ter_or_not],
        'BOND SITES': [bond_atom_indices],
        'MOL': [new_NNAA]
    })

    df = pd.concat([df, new_data], ignore_index=True)

    # deduplicate by SMILES
    df = df.drop_duplicates(subset=['SMILES'])
    
    return df

def add_IDs(df):
    # group df by TAUTOMER HASH
    tautomer_groups = df['TAUTOMER HASH'].drop_duplicates().reset_index(drop=True)
    
    for i, tautomer_hash in enumerate(tautomer_groups):
        df.loc[df['TAUTOMER HASH'] == tautomer_hash, 'ID'] = f"X{i}"

    # if ['TERMINAL'] == 'ter', add 'ter' to the ID
    df.loc[df['TERMINAL'] == 'ter', 'ID'] = df['ID'] + 'ter'

    return df

def relabel_NNAA(mol, NNAA_df):
    visited_Unk_labels, visited_NNAA_labels = [], []
    for atom_idx in range(mol.GetNumAtoms()):
        try:
            label = mol.GetAtomWithIdx(atom_idx).GetProp("AA")
            if label.startswith("Unk") and label not in visited_Unk_labels:
                visited_Unk_labels.append(label)
                rwmol_from_peptide = Chem.RWMol(mol)
                remove_atoms(rwmol_from_peptide, mol, different_NNAA, Unk_label=label)
                for idx, NNAA_row in NNAA_df.iterrows():
                    if perfect_match(rwmol_from_peptide, NNAA_row['MOL']):
                        nnaa_name = NNAA_row['ID']
                        seen_times = visited_NNAA_labels.count(nnaa_name)
                        nnaa_prop = f"{nnaa_name}:{seen_times}"
                        mol = relabel_prop(mol, label, nnaa_prop)
                        visited_NNAA_labels.append(nnaa_name)
                        break
        except:
            continue
    return mol

def different_NNAA(label, Unk_label):
    return label != Unk_label


def relabel_prop(mol, label, nnaa_name):
    for atom_idx in range(mol.GetNumAtoms()):
        try:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.HasProp("AA") and atom.GetProp("AA") == label:
                atom.SetProp("AA", nnaa_name)
        except:
            continue
    return mol


def perfect_match(rwmol_NNAA, nnaa_mol):
    return (
        rwmol_NNAA.HasSubstructMatch(nnaa_mol, useChirality=True)
        and nnaa_mol.GetNumAtoms() == rwmol_NNAA.GetNumAtoms()
    )

def NNAAs_with_OH_removed(NNAA_df):
    new_rows = []  # List to store the new rows

    for _, row in NNAA_df.iterrows():
        mol = row['MOL']
        rwmol_NNAA = Chem.RWMol(mol)
        backbone_indices = rwmol_NNAA.GetSubstructMatches(loose_backbone)
        
        for backbone_index in backbone_indices:
            OH_atom_i = backbone_index[OH_position]
            rwmol_NNAA.GetAtomWithIdx(OH_atom_i).SetProp("ToBeRemoved", "ToBeRemoved")
        
        num_atoms = rwmol_NNAA.GetNumAtoms() - 1
        while num_atoms >= 0:
            if rwmol_NNAA.GetAtomWithIdx(num_atoms).HasProp("ToBeRemoved"):
                rwmol_NNAA.RemoveAtom(num_atoms)
                result_mol = rwmol_NNAA.GetMol()

                # Add a new row to new_rows with the same data except for the modified 'MOL'
                new_row = row.copy()
                new_row['MOL'] = result_mol
                new_rows.append(new_row)

                num_atoms -= 1
            num_atoms -= 1
    
    # Convert new_rows to a DataFrame and concatenate with the original NNAA_df
    new_rows_df = pd.DataFrame(new_rows)
    NNAA_df = pd.concat([NNAA_df, new_rows_df], ignore_index=True)

    return NNAA_df

def remove_small_substructs(mol):
    substructures = Chem.GetMolFrags(mol, asMols=True)
    if len(substructures) <= 1:
        return mol, False
    else:
        error = "Multiple substructures. Removing the smaller ones."
        substructure_sizes = [sub.GetNumAtoms() for sub in substructures]
        largest_substructure_index = substructure_sizes.index(max(substructure_sizes))
        for i in range(len(substructures)):
            if i != largest_substructure_index:
                modified_mol = Chem.DeleteSubstructs(mol, substructures[i])
        return modified_mol, error

def has_unlabelled_atom(mol, seq_list):
    if "?" in seq_list:
        return True
    for atom in mol.GetAtoms():
        if not atom.HasProp("AA"):
            return True
    return False

def linear(peptide_bonds, aminos):
    return len(peptide_bonds) == len(aminos) - 1

def ter_in_the_middle(seq_list):
    for i, amino in enumerate(seq_list):
        if amino.endswith("ter") and i != 0 and i != len(seq_list) - 1:
            return True

def filter_out(seq_list, mol, peptide_bonds):
    if not linear(peptide_bonds, seq_list):
        return "Not linear"
    if has_unlabelled_atom(mol, seq_list):
        return "Has unlabelled atom"
    if ter_in_the_middle(seq_list):
        return "Terminal amino acid in the middle"
    return False

def record_bond_sites(NNAA):
    indices = []
    for atom in NNAA.GetAtoms():
        if atom.HasProp("bond_site"):
            indices.append(atom.GetIdx())
    return indices

def count_aminos(split_seq, NNAA_counts):
    for amino in split_seq:
        # Count the number of times each NNAA is seen in the output sequences
        if amino.startswith("X"):
            if amino in NNAA_counts:
                NNAA_counts[amino] += 1
            else:
                NNAA_counts[amino] = 1
    return NNAA_counts

def load_data(input_file):
    # Load the data
    print("0/4 Loading input data...")
    df = pd.read_csv(input_file, sep='\t', on_bad_lines='warn')

    # Check if the 'ID' column exists
    if 'ID' not in df.columns:
        df['ID'] = range(1, len(df) + 1)  # Create an 'ID' column with unique sequential numbers

    # Check if the 'ISOSMILES' column exists
    if 'ISOSMILES' not in df.columns:
        df['ISOSMILES'] = None  # Create an empty 'ISOSMILES' column if it doesn't exist

    # Check if the 'SMILES' column exists
    if 'SMILES' not in df.columns:
        df['SMILES'] = None  # Create an empty 'SMILES' column if it doesn't exist

    # Determine which column to use for the SMILES
    df['SMILES'] = df['ISOSMILES'].fillna(df['SMILES']).str.strip()

    # Remove rows where both 'ISOSMILES' and 'SMILES' are missing or empty
    df = df[df['SMILES'].ne("")]

    # Drop ISOSMILES column
    df = df.drop(columns=['ISOSMILES'])

    # drop rows where 'SMILES' is empty
    df = df[(df['SMILES'] != '') & (df['SMILES'].notna())]

    # convert to a dataframe
    df = pd.DataFrame(df)
    return df

def process_molecule_batch(batch_df, smi2mol, ignore_cyclic_peptide, min_amino_acids, progress_bar):
    local_mol_data = []

    for mol_index, row in batch_df.iterrows():
        try:
            smi = row['SMILES']
            if not smi:
                local_mol_data.append((mol_index, None, None, "No SMILES provided", None, None))
                continue

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                local_mol_data.append((mol_index, None, None, "Invalid SMILES", None, None))
                continue

            mol, error = remove_small_substructs(mol)
            if error:
                local_mol_data.append((mol_index, None, None, error, None, None))
                continue

            match_AA(mol, smi2mol)
            peptide_bonds = label_peptide_bonds(mol)

            if len(peptide_bonds) < min_amino_acids - 1:
                local_mol_data.append((mol_index, None, None, "Not enough amino acids", None, None))
                continue

            num_NNAAs = label_NNAAs(mol, peptide_bonds)
            all_AA = num_NNAAs == 0

            label_boundary_bonds(mol)

            if mol_is_cyclic_peptide(mol, ignore_cyclic_peptide):
                local_mol_data.append((mol_index, None, None, "Cyclic peptide", None, None))
                continue

            NNAAs_info = []
            if not all_AA:
                NNAAs = get_NNAAs(mol)
                if NNAAs == "error":
                    local_mol_data.append((mol_index, None, None, "Disconnected molecule", None, None))
                    continue
                else:
                    for NNAA in NNAAs:
                        ter_or_not = detect_terminal(NNAA)
                        bond_sites = record_bond_sites(NNAA)
                        NNAAs_info.append((NNAA, ter_or_not, bond_sites))
            
            local_mol_data.append((mol_index, mol, all_AA, None, peptide_bonds, NNAAs_info))

        except:
            local_mol_data.append((mol_index, None, None, "Unknown error", None, None))

    progress_bar.update(1)
    return local_mol_data

def label_molecules_in_batches(mol_df, batch_size, smi2mol, ignore_cyclic_peptide, min_amino_acids, max_workers):
    # Initialize columns and dataframes
    mol_df[['ERROR', 'MOL', 'ALL AA', 'PEPTIDE BONDS']] = ["", "", False, ""]
    NNAA_df = pd.DataFrame(columns=['ID', 'SMILES', 'TERMINAL', 'BOND SITES'])

    indices = list(mol_df.index)
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    futures = []
    progress_bar = tqdm(total=len(indices) // batch_size, desc="1/4 Labelling molecules", leave=True)

    # Use ThreadPoolExecutor for parallel batch processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_indices in batches:
            batch_df = mol_df.loc[batch_indices]
            futures.append(
                executor.submit(process_molecule_batch, batch_df, smi2mol, ignore_cyclic_peptide, min_amino_acids, progress_bar)
            )

    progress_bar.close()

    with tqdm(total=len(mol_df), desc="2/4 Storing NNAAs") as pbar:
            for future in as_completed(futures):
                batch_results = future.result()

                for mol_index, mol, all_AA, error, peptide_bonds, NNAAs_info in batch_results:
                    if mol is None:
                        mol_df.at[mol_index, 'ERROR'] = error
                        continue

                    mol_df.at[mol_index, 'MOL'] = mol
                    mol_df.at[mol_index, 'ALL AA'] = all_AA
                    mol_df.at[mol_index, 'PEPTIDE BONDS'] = peptide_bonds

                    if NNAAs_info:
                        for NNAA, ter_or_not, bond_sites in NNAAs_info:
                            NNAA_df = enlist_NNAA(NNAA, NNAA_df, ter_or_not, bond_sites)
                
                pbar.update(len(batch_results))

    return NNAA_df, mol_df

def highlight_bonds_with_AA(mol_s):  # with AA colors
    bond_highlights = defaultdict(lambda: [])
    for bond in mol_s.GetBonds():
        atom1_i, atom2_i, prop1, prop2 = get_connected_atoms_and_props(bond, mol_s)
        if (label_belongs_to_AA(prop1) and prop1 == prop2):  # if the bond is within the same AA
            bond_highlights[bond.GetIdx()].append(aa2color_dict[prop1[:3]])
    return bond_highlights


def relabel_batch(mol_df, NNAA_df):
    # Initialize a list to collect row data
    local_mol_data = []

    for _, row in mol_df.iterrows():
        mol_index = row['ID']
        mol = row['MOL']
        all_AA = row['ALL AA']
        peptide_bonds = row['PEPTIDE BONDS']

        try:
            # Process molecule if not all amino acids are labeled
            if not all_AA:
                mol = relabel_NNAA(mol, NNAA_df)
            
            # Perform various processing tasks
            bond_highlights = highlight_bonds_with_AA(mol)
            peptide_bonded_props, peptide_bonded_atoms = search_terminal_AA(mol)
            first_atom_index = get_first_atom_index(mol, peptide_bonded_props, peptide_bonded_atoms)
            aa_list = reorder_AAs(mol, first_atom_index)
            split_seq = write_seq(aa_list)
            seq = "".join(split_seq)

            error = filter_out(split_seq, mol, peptide_bonds)

            if error:
                seq = ""

        except Exception as e:
            error = str(e)  # Ensure error is a string
            seq = ""

        # Collect data in a list of dictionaries
        local_mol_data.append({'ID': mol_index, 'SEQUENCE': seq, 'ERROR': error, 'BOND HIGHLIGHTS': bond_highlights})

    return pd.DataFrame(local_mol_data)

def relabel_batches(mol_df, NNAA_df, batch_size):
    # Check if NNAA_df is empty
    if NNAA_df.empty:
        print("Warning: NNAA_df is empty. No NNAAs to process.")

    # Ensure NNAA_df has an index
    if NNAA_df.index.empty:
        NNAA_df = NNAA_df.reset_index(drop=True)
    mol_df['BOND HIGHLIGHTS'] = ""
    mol_df_copy = mol_df[mol_df['MOL'] != ""].copy()
    indices = list(mol_df_copy.index)

    def process_batch(batch_indices):
        batch_df = mol_df_copy.loc[batch_indices]
        return relabel_batch(batch_df, NNAA_df)

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            futures.append(executor.submit(process_batch, batch_indices))

        local_mol_df = mol_df.copy()

        for future in tqdm(as_completed(futures), total=len(futures), desc="4/4 Relabelling mols"):
            mol_dataset_per_batch = future.result()

            for _, row in mol_dataset_per_batch.iterrows():
                local_mol_df.loc[local_mol_df['ID'] == row['ID'], ['SEQUENCE', 'ERROR', 'BOND HIGHLIGHTS']] = row[['SEQUENCE', 'ERROR', 'BOND HIGHLIGHTS']].values

    return local_mol_df



def output_NNAA(NNAA_df, output_dir):
    # Drop the 'MOL' column
    NNAA_df = NNAA_df.drop(columns=['MOL'])
    NNAA_df['TAUTOMERS'] = None

    # Add 'COUNT' by 'TAUTOMER HASH' group and deduplicate by 'TAUTOMER HASH'
    NNAA_df = NNAA_df.groupby('TAUTOMER HASH').agg(
        ID=('ID', 'first'),
        SMILES=('SMILES', 'first'),
        TAUTOMERS=('SMILES', lambda x: ','.join(x.unique())),
        TERMINAL=('TERMINAL', 'first'),
        BOND_SITES=('BOND SITES', 'first'),
    ).reset_index().drop_duplicates(subset='TAUTOMER HASH', keep='first')

    NNAA_df = NNAA_df.drop(columns=['TAUTOMER HASH'])

    print(output_dir)

    NNAA_df.to_csv(os.path.join(output_dir, "raw/ncAAs_raw.txt"), sep='\t', index=False)


def output_mols(mol_df, output_dir, draw):
    if draw:
        drawer = MoleculeDrawer(output_dir)
        
        def safe_draw(row):
            try:
                drawer.draw_input_mol(row['MOL'], row['ID'], row['SEQUENCE'], row['BOND HIGHLIGHTS'])
            except Exception as e:
                return None  # Return None to effectively ignore this row
        
        # Apply the safe drawing function to each row
        mol_df.apply(lambda row: safe_draw(row), axis=1)

    mol_df.drop(columns=['MOL', 'PEPTIDE BONDS'], inplace=True)

    # bring 'SEQUENCE' column next to 'ID'
    cols = ['ID', 'SEQUENCE'] + [col for col in mol_df.columns if col not in ['ID', 'SEQUENCE']]
    mol_df = mol_df[cols]

    mol_df.to_csv(os.path.join(output_dir, "raw/sequences_raw.txt"), sep='\t', index=False)

def get_rdkit_tautomer_hash(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    layers = RegistrationHash.GetMolLayers(mol)
    return layers[HashLayer.TAUTOMER_HASH]

def main():
    mol_df = load_data(input_file)
    NNAA_df, mol_df = label_molecules_in_batches(mol_df, batch_size, smi2mol, ignore_cyclic_peptide, min_amino_acids, max_workers)
    NNAA_df['TAUTOMER HASH'] = NNAA_df['SMILES'].apply(get_rdkit_tautomer_hash)
    NNAA_df = NNAAs_with_OH_removed(NNAA_df)
    NNAA_df = add_IDs(NNAA_df)
    mol_df = relabel_batches(mol_df, NNAA_df, batch_size)
    output_NNAA(NNAA_df, output_dir)
    output_mols(mol_df, output_dir, draw)

if __name__ == '__main__':
    args = parse_arguments()

    input_file = args.input_file
    ignore_cyclic_peptide = not args.process_cyclic
    min_amino_acids = args.min_amino_acids
    batch_size = args.batch_size
    output_dir = args.output_dir
    max_workers = args.max_workers
    draw = args.draw

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)

    main()
