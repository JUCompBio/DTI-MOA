import os
from rdkit.Chem import AllChem
from rdkit import Chem
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data


allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)),
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    "possible_chirality_list": [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER],
    "possible_hybridization_list": [Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "possible_implicit_valence_list": [0, 1, 2, 3, 4, 5, 6],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "possible_bonds": [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC],
    "possible_bond_dirs": [Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT, Chem.rdchem.BondDir.ENDDOWNRIGHT],  # only for double bond stereo information
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features["possible_atomic_num_list"].index(atom.GetAtomicNum())] + [allowable_features["possible_chirality_list"].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features["possible_bonds"].index(bond.GetBondType())] + [allowable_features["possible_bond_dirs"].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class MOADataset(Dataset):
    def __init__(self, df, smiles_encoding_root, gnn_encoding_root, prot_encoding_root, dssp_root, num_classes=8):
        """
        df: Dataframe containing the following columns -
        DrugID (str), Drug (str), DrugEncodingFile (str), TargetID (str), Target (str: Optional), TargetEncodingFile (str), TargetDSSPFile (str), Y (int/float)
        """
        super().__init__()
        self.df = df
        self.smiles_encoding_root = smiles_encoding_root
        self.gnn_encoding_root = gnn_encoding_root
        self.prot_encoding_root = prot_encoding_root
        self.dssp_root = dssp_root
        self.num_classes = num_classes

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        target_enc = torch.load(os.path.join(self.prot_encoding_root, row["TargetEncodingFile"]), "cpu").to(torch.float32)
        target_dssp = torch.load(os.path.join(self.dssp_root, row["TargetDSSPFile"]), "cpu").to(torch.float32)
        drug_enc = torch.load(os.path.join(self.smiles_encoding_root, row["DrugEncodingFile"]), "cpu").to(torch.float32)
        drug_graph = torch.load(os.path.join(self.gnn_encoding_root, row["DrugEncodingFile"]), "cpu").to(torch.float32)
        y = torch.tensor(row["Y"], dtype=torch.float32)
        if self.num_classes > 2:
            y = F.one_hot(y.to(torch.int64), self.num_classes).to(torch.float32)
        return target_enc, target_dssp, drug_enc, drug_graph, y
