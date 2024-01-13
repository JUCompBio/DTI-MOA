import os
from tqdm.auto import tqdm
import pandas as pd
import random
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
df.to_csv(os.path.join(dirpath, "../data/drug-target-moa-mod.csv"), index=False)
