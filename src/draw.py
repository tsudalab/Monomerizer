import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np

class MoleculeDrawer:
    def __init__(self, output_dir="output/tmp"):
        self.output_dir = os.path.join(output_dir, "raw/images")
        os.makedirs(self.output_dir, exist_ok=True)
        self.aa2color_dict = {
            "Asp": (0.902, 0.039, 0.039), "Glu": (0.961, 0.1, 0.537), "Arg": (0.078, 0.353, 1), "Lys": (0.42, 0.353, 1),
            "His": (0.51, 0.51, 0.824), "Tyr": (0.196, 0.196, 0.667), "Phe": (0.341, 0.196, 0.667), "Trp": (0.706, 0.353, 0.706),
            "Asn": (0, 0.863, 0.863), "Gln": (0.5, 0.82, 0.863), "Met": (0.902, 0.902, 0), "Cys": (0.722, 0.902, 0),
            "Ser": (0.98, 0.588, 0), "Thr": (0, 0.612, 0.412), "Gly": (0.98, 0.922, 0.922), "Ala": (0.784, 0.784, 0.639),
            "Val": (0.059, 0.51, 0.059), "Leu": (0.29, 0.51, 0.059), "Ile": (0.29, 0.51, 0.471), "Pro": (1, 0.588, 0.51)
        }
    
    def sort_atom_highlights(self, mol):
        atom_highlights = defaultdict(list)
        for atom_idx in range(mol.GetNumAtoms()):
            labelled_atom = mol.GetAtomWithIdx(atom_idx)
            AA_label = labelled_atom.GetProp("AA")
            if self.label_belongs_to_AA(AA_label):
                three_letter_label = AA_label[:3]
                atom_highlights[atom_idx].append(self.aa2color_dict[three_letter_label])

        # Convert defaultdict to dict of lists
        return {k: list(v) for k, v in atom_highlights.items()}

    def create_colormap(self):
        legend_data = [(aa[:3], color) for aa, color in self.aa2color_dict.items() if aa != "Unk"]
        fig, ax = plt.subplots(figsize=(1, 1))
        cmap = ListedColormap([color for _, color in legend_data])
        cax = ax.matshow(np.arange(len(legend_data)).reshape(1, -1), cmap=cmap)
        cbar = fig.colorbar(cax, ticks=np.arange(len(legend_data)), aspect=5)
        cbar.set_ticklabels([label for label, _ in legend_data])
        cbar.ax.tick_params(labelsize=3)
        ax.axis("off")
        plt.savefig(os.path.join(self.output_dir, "colormap.png"), bbox_inches="tight", dpi=300)
        plt.close()
    
    def draw_input_mol(self, mol, mol_index, seq, bond_highlights):
        atom_highlights = self.sort_atom_highlights(mol)

        # Ensure bond_highlights is a dict of lists
        bond_highlights = {k: list(v) for k, v in bond_highlights.items()} if bond_highlights else {}

        mol_name = f"mol_{mol_index}"
        legend = f'{mol_name}\nseq: {seq}\n{"8< = peptide bond"}\nAA_NAME:SEEN_COUNT:SEQUENCE_POSITION\n'

        self.draw_mol(mol, atom_highlights, bond_highlights, legend, mol_name)
        self.create_colormap()

    
    def draw_mol(self, mol, atom_highlights, bond_highlights, legend, mol_name):
        view = rdMolDraw2D.MolDraw2DSVG(600, 300)
        view.drawOptions().useBWAtomPalette()
        view.DrawMoleculeWithHighlights(mol, legend, dict(atom_highlights), dict(bond_highlights), {}, {})
        view.FinishDrawing()
        with open(os.path.join(self.output_dir, f"{mol_name}.svg"), "w") as f:
            f.write(view.GetDrawingText())

    def label_belongs_to_AA(self, label):
        shorter_label = label[:3]
        return shorter_label != "Unk" and not label.startswith("X")