#!/usr/bin/env python3

import re, ast
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Parse the input arguments
parser = argparse.ArgumentParser(description="Preprocess the generated sequences file")
parser.add_argument("--sequence_file", type=str, help="Path to the generated sequences file", default="sequences_generated.txt")
parser.add_argument("--NNAA_file", type=str, help="Path to the NNAA file", default="dictionary.txt")
parser.add_argument("--batch_size", type=int, help="Batch size for processing sequences", default=8)
parser.add_argument("--output_dir", type=str, help="Output directory", default="output")
parser.add_argument("--demonomerized_file", type=str, help="Output demonomerized file name", default="demonomerized.txt")

args = parser.parse_args()

valid_backbone = Chem.MolFromSmarts("[NH,NH2]CC(=O)")
valid_backbone_OH = Chem.MolFromSmarts("[NH,NH2]CC(=O)O")
peptide_bond_mol = Chem.MolFromSmarts("[N,n][C,c]C(=O)[*!O]") # [*!O] ensures it does not match AAter
edge_C = 2
edge_N = 0
edge_O = 4

name_smi_dict = {
    # isomeric SMILES from pubchem. eg https://pubchem.ncbi.nlm.nih.gov/compound/Alanine except for Asp (from https://www.guidetopharmacology.org/GRAC/LigandDisplayForward?ligandId=3309) and Arg (from https://en.wikipedia.org/wiki/Arginine)
    "Wter": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O)O)N",
    "W": "C1=CC=C2C(=C1)C(=CN2)C[C@@H](C(=O))N",
    "Rter": "C(C[C@@H](C(=O)O)N)CNC(=N)N",
    "R": "C(C[C@@H](C(=O))N)CNC(=N)N",
    "Hter": "C1=C(NC=N1)C[C@@H](C(=O)O)N",
    "H": "C1=C(NC=N1)C[C@@H](C(=O))N",
    "Pter": "C1C[C@H](NC1)C(=O)O",
    "P": "C1C[C@H](NC1)C(=O)",
    "Kter": "C(CCN)C[C@@H](C(=O)O)N",
    "K": "C(CCN)C[C@@H](C(=O))N",
    "Mter": "CSCC[C@@H](C(=O)O)N",
    "M": "CSCC[C@@H](C(=O))N",
    "Qter": "C(CC(=O)N)[C@@H](C(=O)O)N",
    "Q": "C(CC(=O)N)[C@@H](C(=O))N",
    "Nter": "C([C@@H](C(=O)O)N)C(=O)N",
    "N": "C([C@@H](C(=O))N)C(=O)N",
    "Eter": "C(CC(=O)O)[C@@H](C(=O)O)N",
    "E": "C(CC(=O)O)[C@@H](C(=O))N",
    "Dter": "OC(=O)C[C@@H](C(=O)O)N",
    "D": "OC(=O)C[C@@H](C(=O))N",
    "Yter": "C1=CC(=CC=C1C[C@@H](C(=O)O)N)O",
    "Y": "C1=CC(=CC=C1C[C@@H](C(=O))N)O",
    "Fter": "C1=CC=C(C=C1)C[C@@H](C(=O)O)N",
    "F": "C1=CC=C(C=C1)C[C@@H](C(=O))N",
    "Iter": "CC[C@H](C)[C@@H](C(=O)O)N",  # TODO add correct hydroxyl oxygen for every AA terminal
    "I": "CC[C@H](C)[C@@H](C(=O))N",
    "Lter": "CC(C)C[C@@H](C(=O)O)N",
    "L": "CC(C)C[C@@H](C(=O))N",
    "Vter": "CC(C)[C@@H](C(=O)O)N",
    "V": "CC(C)[C@@H](C(=O))N",
    "Tter": "C[C@H]([C@@H](C(=O)O)N)O",
    "T": "C[C@H]([C@@H](C(=O))N)O",
    "Cter": "C([C@@H](C(=O)O)N)S",
    "C": "C([C@@H](C(=O))N)S",
    "Ster": "C([C@@H](C(=O)O)N)O",
    "S": "C([C@@H](C(=O))N)O",
    "Ater": "C[C@@H](C(=O)O)N",
    "A": "C[C@@H](C(=O))N",
    "Gter": "C(C(=O)O)N",
    "G": "C(C(=O))N",
}

def mark_edge(amino, pattern, edge_position):
    matched_indices = amino.GetSubstructMatch(pattern)
    edge_position = matched_indices[edge_position]
    edge_atom = amino.GetAtomWithIdx(edge_position)
    edge_atom.SetProp("atomNote", "edge")
    return edge_atom

def mark_edge_NNAA(NNAA, bond_sites):
    try:
        for i in bond_sites:
            integer = int(i)
            atom = NNAA.GetAtomWithIdx(integer)
            atom.SetProp("atomNote", "edge")
    except:
        print("No bond sites")
        pass

def mark_bond_site(mol, index, symbol):
    for atom in mol.GetAtoms():
        if atom.HasProp("atomNote") and atom.GetSymbol() == symbol:
            atom.SetProp("atomNote", str(index))

def clear_props(atom1, atom2):
    atom1.ClearProp("atomNote")
    atom2.ClearProp("atomNote")

def get_amino_mol(amino_name, name_smi_dict, NNAA_file):
    for aa_name, aa_smi in name_smi_dict.items():
        if aa_name == amino_name:
            amino_mol = Chem.MolFromSmiles(aa_smi)
            try:
                mark_edge(amino_mol, valid_backbone, edge_N)
                mark_edge(amino_mol, valid_backbone, edge_C)
            except:
                for index, row in NNAA_file.iterrows():
                    name = row["ID"]
                    if name == amino_name:
                        bond_info = ast.literal_eval(row["Bond sites"])
                        smiles_rootedAtAtom0 = bond_info[0]
                        bond_sites = bond_info[1:]
                        amino_mol = Chem.MolFromSmiles(smiles_rootedAtAtom0)
                        mark_edge_NNAA(amino_mol, bond_sites)
            return amino_mol
        

def process_batch(batch_df):
    results = []
    for index, row in batch_df.iterrows():
        result_index, result_smiles = process_row(index, row)
        results.append((result_index, result_smiles))
    return results

def process_row(index, row):
    if "SMILES" not in row or type(row["SMILES"]) == float or len(row["SMILES"]) == 0:
        seq = row["SEQUENCE"]
        split_seq = regex.findall(seq)
        ordered_aminos = []
        
        try:
            for alphabet in split_seq:
                amino_mol = get_amino_mol(alphabet, name_smi_dict, NNAA_file)
                ordered_aminos.append(amino_mol)

            # Replace the last amino with the terminal amino
            amino_ter = split_seq[-1]
            if not "ter" in amino_ter and not amino_ter.startswith("Z"):
                amino_ter = f"{amino_ter}ter"
            last_mol = get_amino_mol(amino_ter, name_smi_dict, NNAA_file)
            ordered_aminos[-1] = last_mol

            combined = ordered_aminos[0]
            for i in range(len(ordered_aminos)-1):
                mark_bond_site(combined, i, "C")
                next_amino = ordered_aminos[i+1]
                mark_bond_site(next_amino, i+1, "N")
                combined = Chem.CombineMols(combined, next_amino)
                rwmol = Chem.RWMol(combined)
                for atom1 in rwmol.GetAtoms():
                    if atom1.HasProp("atomNote") and atom1.GetProp("atomNote") == f"{i}":
                        for atom2 in rwmol.GetAtoms():
                            if atom2.HasProp("atomNote") and atom2.GetProp("atomNote") == f"{i+1}":
                                rwmol.AddBond(atom1.GetIdx(), atom2.GetIdx(), Chem.BondType.SINGLE)
                                clear_props(atom1, atom2)
                                if len(rwmol.GetSubstructMatches(peptide_bond_mol)) == i+1:
                                    combined = rwmol.GetMol()
                                    break

            result = Chem.MolToSmiles(combined, isomericSmiles=True, rootedAtAtom=0, canonical=True)
            if '.' in result:
                return index, None  # Indicates unbound atoms
            return index, result
        
        except Exception as e:
            print(f"Error in sequence: {seq}")
            return index, None

    return index, row.get("SMILES")  # Return the existing SMILES if present

NNAA_file = pd.read_csv(args.NNAA_file, sep="\t")
for index, row in NNAA_file.iterrows():
    smiles = row["SMILES"]
    name = row["ID"]
    NNAA = Chem.MolFromSmiles(smiles)
    if NNAA.HasSubstructMatch(valid_backbone_OH):
        rwmol = Chem.RWMol(NNAA)
        OH_i = NNAA.GetSubstructMatch(valid_backbone_OH)[edge_O]
        rwmol.RemoveAtom(OH_i)
        noOH_smiles = Chem.MolToSmiles(rwmol)
        name_smi_dict[name] = noOH_smiles
    if not name.startswith("Z"):
        name = f"{name}ter"
    name_smi_dict[name] = smiles

tokenizer = r"X\d+|Z\d+|[A-WY]"
regex = re.compile(tokenizer)

df = pd.read_csv(args.sequence_file, sep="\t")

# add a column for SMILES
df["SMILES"] = ""

# Process in batches
batch_size = args.batch_size
batches = [df[i:i + batch_size] for i in range(0, df.shape[0], batch_size)]

# Process batches in parallel
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_batch, batch): batch for batch in batches}
    for future in tqdm(as_completed(futures), total=len(futures)):
        results = future.result()
        for index, smiles in results:
            if smiles:
                df.at[index, "SMILES"] = smiles

# ✅ Use output_dir in your logic
os.makedirs(args.output_dir, exist_ok=True)
output_file = os.path.join(args.output_dir, args.demonomerized_file)

# Assuming `df` is your final DataFrame
df.to_csv(output_file, sep="\t", index=False)
