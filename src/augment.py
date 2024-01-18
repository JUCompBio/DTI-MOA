import sys
import argparse
import os
import json
import pandas as pd
import random
import numpy as np
from rdkit import Chem

seed = 42
random.seed(seed)
dirpath = os.path.dirname(__file__)


def randomize_smiles(smiles):
    """Perform a randomization of a SMILES string must be RDKit sanitizable"""
    m = Chem.MolFromSmiles(smiles)
    ans = list(range(m.GetNumAtoms()))
    random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    return Chem.MolToSmiles(nm, canonical=False, isomericSmiles=True)


def get_blosum_prot_list():
    """
    List of proteins that are similar to each other according to BLOSUM
    """
    with open(os.path.join(os.path.dirname(__file__), "../data/blosum.json"), "r") as f:
        blosum = json.load(f)
    blosum_prot_list = {}
    for i in blosum:
        blosum_prot_list[i] = []
        for j in blosum[i]:
            if blosum[i][j] > 0 and i != j:
                blosum_prot_list[i].extend([j] * blosum[i][j])

    return blosum_prot_list


def augment_aa_seq(aa, max_aa_change_ratio):
    blosum_prot_list = get_blosum_prot_list()
    aa = list(aa)
    new_prot = aa.copy()
    for residue_idx in random.sample(list(range(len(aa))), random.choice(list(range(1, int(len(aa) * max_aa_change_ratio + 1) + 1)))):
        if new_prot[residue_idx] in blosum_prot_list and len(blosum_prot_list[new_prot[residue_idx]]):
            new_prot[residue_idx] = random.choice(blosum_prot_list[new_prot[residue_idx]])
    return "".join(new_prot)


def augment_aa_seq_num_and_create_rows(row, idx: int, id_idx: int, max_aa_change_ratio, num, max_tries=10):
    """
    Repeatedly augment same aa seq num times
    """
    aa = row[idx]
    new_aa_set = set([aa])
    try_num = 0
    while len(new_aa_set) <= num:
        new_aa = augment_aa_seq(aa, max_aa_change_ratio)
        if new_aa in new_aa_set:
            try_num += 1
            if try_num > max_tries:
                break
        else:
            try_num = 0
            new_aa_set.add(new_aa)

    new_aa_set.remove(aa)
    if isinstance(row, list) or isinstance(row, np.ndarray):
        new_rows = []
        for i, aa in enumerate(new_aa_set):
            new_row = row.copy()
            new_row[id_idx] += f"_{i+1}"
            new_row[idx] = aa
            new_rows.append(new_row)
        return new_rows
    return new_aa_set


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-path", type=str, required=True, help="Path to DTI-MOA csv df")
    parser.add_argument("--min-action-num", type=int, default=2000, help="Min. no. of drug-targat pairs required for a balanced dataset")
    parser.add_argument("--max-prot-change-ratio", type=float, default=0.01, help="Max. no. of augmenations in a single amino acid sequence")
    args = parser.parse_args(args)

    df = pd.read_csv(args.df_path)
    # Remove inhibitor to make a more balanced dataset
    l = df.loc[df["action"] == "inhibitor"].index.tolist()
    l_to_drop = random.sample(l, k=2000)
    df.loc[df.index.isin(l_to_drop)].to_csv(os.path.join(dirpath, "../data/dropped_inhibitor_moa.csv"), index=False)
    df = df.drop(labels=l_to_drop)

    s = df.groupby(["action"]).size()
    actions_aug = list(map(lambda z: z[0], filter(lambda z: True if s[z[0]] < args.min_action_num and z[0] != "other" else False, s.items())))

    # Augment Targets
    new_rows = []
    tries = 10
    for action in actions_aug:
        num_available = len(df[df["action"] == action])
        arr = [0] * num_available
        # Set number of augmentations to be done for each of the available proteins
        for _ in range(args.min_action_num - num_available):
            arr[random.randint(0, num_available - 1)] += 1
        # For each row, use the given prot to make num new ones
        for num, row in zip(arr, df[df["action"] == action].values):
            if num > 0:
                new_rows.extend(augment_aa_seq_num_and_create_rows(row, -2, 2, args.max_prot_change_ratio, num, tries))

    # Temp variable just to say the whole dataset again to augment for drugs
    new_df_ = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
    new_df_ = new_df_.loc[new_df_["action"].isin(actions_aug)].values.tolist()

    # Augment the drug (SMILES or Protein)
    for row in random.sample(new_df_, int(len(new_rows))):
        if row[7] == "SMILES":
            for i in range(random.choice([0, 1, 1, 2, 2])):
                new_row = row.copy()
                new_row[6] = randomize_smiles(new_row[6])
                new_row[0] += f"_{i}"
                new_rows.append(new_row)
        else:
            new_rows.extend(augment_aa_seq_num_and_create_rows(row, 6, 0, args.max_prot_change_ratio, random.choice([0, 1, 1, 2, 2]), tries))

    print("Number of rows added:", len(new_rows))
    df = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
    df.to_csv(os.path.join(dirpath, "../data/drug-target-moa-aug.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
