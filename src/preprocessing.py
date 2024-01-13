import re
import numpy as np
import os
import json
import requests
import urllib
from tqdm.auto import tqdm
import pandas as pd
import random
from rdkit import Chem
import gzip
import shutil
import preprocessing_utils

seed = 42
random.seed(seed)
dirpath = os.path.dirname(__file__)

# Read initial DTI-MOA CSV file
df = pd.read_csv(os.path.join(dirpath, "../data/drug-target-moa.csv"))
# Accumulate related MOA types/subdivisions
df["action"].replace(["blocker", "downregulator", "inactivator", "translocation inhibitor", "weak inhibitor"], "inhibitor", inplace=True)
df["action"].replace(["binder", "binding"], "ligand", inplace=True)
df["action"].replace(["inhibitory allosteric modulator", "negative modulator", "positive allosteric modulator", "modulator"], "allosteric modulator", inplace=True)
df["action"].replace(["partial antagonist"], "antagonist", inplace=True)
df["action"].replace(["inverse agonist", "partial agonist"], "agonist", inplace=True)
df["action"].replace(["activator", "stimulator", "potentiator"], "inducer", inplace=True)
# Keep only important MOA that occur a substantial number of times
df = df.loc[df["action"].isin(["inhibitor", "ligand", "substrate", "allosteric modulator", "antagonist", "agonist", "inducer"])]
df = df[df["uniprotkb-id"] != "Q8WXI7"]  # Too long
df = df[df["structure"].str.len() != 748]  # SMILE tokenizer sequence larger than 512
df = df.drop_duplicates()
# Keep only one MOA per drug-target pair
# MOA types such as ligand or substrate are discarded when other more specific MOA is available for a given drug target pair
df_ = df.groupby(["drugbank-id", "drug-type", "uniprotkb-id", "uniprotac-id", "gene-name", "structure", "type", "aa-seq"])["action"].apply(set).reset_index()
for i, row in df_.iterrows():
    if len(row["action"]) == 1:
        df_.iat[i, -1] = row["action"].pop()
    else:
        if len(row["action"] - {"ligand"}) == 1:
            df_.iat[i, -1] = (row["action"] - {"ligand"}).pop()
        elif len(row["action"] - {"substrate"}) == 1:
            df_.iat[i, -1] = (row["action"] - {"substrate"}).pop()
        elif len(row["action"] - {"ligand", "substrate"}) == 1:
            df_.iat[i, -1] = (row["action"] - {"ligand", "substrate"}).pop()
        else:
            df_.iat[i, -1] = random.choice(list(row["action"]))
df = df_[df.columns]

# Perform DSSP for all proteins
uniprot_ids = df["uniprotkb-id"].unique().tolist()
db_ids = df.loc[df["type"] == "polypeptide"]["drugbank-id"].unique().tolist()
mapping = []

os.makedirs(os.path.join(dirpath, "../data/pdb"), exist_ok=True)
os.makedirs(os.path.join(dirpath, "../data/dssp"), exist_ok=True)

for uid in tqdm(uniprot_ids):
    r = preprocessing_utils.uniprot_to_dssp(uid)
    mapping.append([None, uid, r])

for dbid in tqdm(db_ids):
    uid = preprocessing_utils.get_uid_from_dbid(dbid)
    mapping.append([dbid, uid, preprocessing_utils.uniprot_to_dssp(uid)])

pd.DataFrame(mapping, columns=["DrugBankID", "UniProtID", "path"]).to_csv(os.path.join(dirpath, "../data/mapping.csv"), index=False)

# Add drug-target pairs that do not interact based on atc (New class "unknown" formed)
atc_df = pd.read_csv(os.path.join(dirpath, "../data/drugs_atc.csv"))
atc_df.rename(columns={"drugbank_id": "drugbank-id"}, inplace=True)
atc_df["atc"] = atc_df["atc"].str[0]
atc_set = set(atc_df["atc"].drop_duplicates().values)
atc_df = atc_df.drop_duplicates()
atc_merged_df = pd.merge(df[["drugbank-id", "uniprotkb-id"]], atc_df, "left", on="drugbank-id")
# Generate no interaction DTI pairs
rows = []
for i in tqdm(range(4000)):
    rand_drug = random.choice(df["drugbank-id"].drop_duplicates().values)
    atcs = atc_merged_df.loc[atc_merged_df["drugbank-id"] == rand_drug]["atc"].drop_duplicates().values
    rem_atcs = atc_set - set(atcs)
    rem_targets = atc_merged_df[["uniprotkb-id", "atc"]].loc[atc_merged_df["atc"].isin(rem_atcs)]["uniprotkb-id"].drop_duplicates().values
    rand_target = random.choice(rem_targets)
    rows.append((rand_drug, None, rand_target, None, None, "unknown", *random.choice(df.loc[df["drugbank-id"] == rand_drug][["structure", "type"]].values), df.loc[df["uniprotkb-id"] == rand_target]["aa-seq"].values[0]))

df = pd.concat([df, pd.DataFrame(set(rows), columns=df.columns)], ignore_index=True)

num_classes = len(df["action"].unique())
print("Number of classes:", num_classes)
print("Classes:", df["action"].unique())
output_map = {df.groupby(["action"]).size().index[k]: k for k in range(num_classes)}
df["output"] = [output_map[i] for i in df["action"]]
print("Number of rows:", len(df))

# Remove inhibitor to make a more balanced dataset
l = df.loc[df["action"] == "inhibitor"].index.tolist()
l_to_drop = random.sample(l, k=2000)
df.loc[df.index.isin(l_to_drop)].to_csv(os.path.join(dirpath, "../data/dropped_inhibitor_moa.csv"), index=False)
df = df.drop(labels=l_to_drop)
df.groupby(["action"]).size()

# Minimum number of MOA required in the df for balance. If not present, then augment.
min_action_num = 2000
max_prot_change_ratio = 0.01
s = df.groupby(["action"]).size()
actions_aug = list(map(lambda z: z[0], filter(lambda z: True if s[z[0]] < min_action_num and z[0] != "other" else False, s.items())))

# Augment Targets
new_rows = []
tries = 10
for action in actions_aug:
    num_available = len(df[df["action"] == action])
    arr = [0] * num_available
    # Set number of augmentations to be done for each of the available proteins
    for _ in range(min_action_num - num_available):
        arr[random.randint(0, num_available - 1)] += 1
    # For each row, use the given prot to make num new ones
    for num, row in zip(arr, df[df["action"] == action].values):
        new_rows.extend(preprocessing_utils.augment_aa_seq_num_and_create_rows(row[-2], max_prot_change_ratio, num, row, tries))

# Temp variable just to say the whole dataset again to augment for drugs
new_df_ = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
new_df_ = new_df_.loc[new_df_["action"].isin(actions_aug)].values.tolist()

# Augment the drug (SMILES or Protein)
for row in random.sample(new_df_, int(len(new_rows) / 2)):
    if row[7] == "SMILES":
        for i in range(random.choice(list(range(2)))):
            new_row = row.copy()
            new_row[6] = preprocessing_utils.randomize_smiles(new_row[6])
            new_row[0] += f"_{i}"
            new_rows.append(new_row)
    else:
        new_rows.extend(preprocessing_utils.augment_aa_seq_num_and_create_rows(row[6], max_prot_change_ratio, num, row, tries))

df = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
df.to_csv(os.path.join(dirpath, "../data/drug-target-moa-aug.csv"), index=False)
