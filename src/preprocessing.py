import os
from tqdm.auto import tqdm
import pandas as pd
import random
import re
import os
import json
import requests
import urllib
import random
import gzip
import shutil
from DSSPparser import parseDSSP


def download_alphafold_pdb_file(uid: str, root: str):
    """
    Downloads Alphafold prediction file from web based on uniprot ID (uid).
    """
    try:
        urllib.request.urlretrieve(f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.pdb", os.path.join(root, f"{uid}.pdb"))
        return os.path.join(root, f"{uid}.pdb")
    except Exception:
        return False


def get_pdb_from_uid(uid: str):
    """
    Fetches (from web) a json file from UniprotDB based on ID (uid). Checks if corresponding pdb id can be obtained.
    """
    uniprot_data = json.loads(requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.json").content)
    # Multiple structures possible for the same protein sequence
    pdb_id_list = []
    if "uniProtKBCrossReferences" in uniprot_data:
        for db in uniprot_data["uniProtKBCrossReferences"]:
            if "pdb" in db["database"].lower():
                pdb_id_list.append(db["id"])
    return pdb_id_list


def download_pdb_pdb_file(pid: str, root: str):
    """
    Downloads pdb file for corresponding pid from RCSB
    root: dir to download pdb file
    """
    f1 = os.path.join(root, f"{pid}.pdb.gz")
    f2 = os.path.join(root, f"{pid}.pdb")
    try:
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.pdb.gz", f1)
        with gzip.open(f1, "rb") as f_in:
            with open(f2, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(f1)
        return f2
    except:
        return False


def perform_dssp(root_inp: str, root_out: str, pid: str):
    """
    Performs DSSP to get the secondary structure information from PDB file (pid)
    """
    f1 = os.path.join(root_inp, f"{pid}.pdb")
    f2 = os.path.join(root_out, f"{pid}.dssp")
    os.system(f"dssp -i {f1} -o {f2}")
    return f2


def uniprot_to_dssp(uid, seq, try_pdb=False):
    """
    Function to get secondary structure information for a given uniprot ID (uid)
    """
    dirpath = os.path.dirname(__file__)
    # Check if Alphafold already has a 3D prediction for the given uniprot id
    pdb_path = download_alphafold_pdb_file(uid, os.path.join(dirpath, "../data/pdb/"))
    if pdb_path:
        dssp_path = perform_dssp(os.path.join(dirpath, "../data/pdb/"), os.path.join(dirpath, "../data/dssp/"), uid)
        if os.path.exists(dssp_path):
            r = parse_dssp_check(dssp_path, seq)
            if isinstance(r, bool) and r:
                return os.path.basename(dssp_path), None
            elif isinstance(r, str):
                print(f"DSSP for uid: {uid} non-matching corresponding aa-seq.")
                return os.path.basename(dssp_path), r
            else:
                print(f"DSSP for uid: {uid} could not be generated.")
        else:
            print(f"DSSP for uid: {uid} could not be generated.")
    else:
        print(f"Alphafold pdb for uid: {uid} not found.")
        if try_pdb:
            print("Trying RCSB database...")
            for pdb_id in get_pdb_from_uid(uid):
                pdb_path = download_pdb_pdb_file(pdb_id, os.path.join(dirpath, "../data/pdb/"))
                if pdb_path:
                    dssp_path = perform_dssp(os.path.join(dirpath, "../data/pdb/"), os.path.join(dirpath, "../data/dssp/"), pdb_id)
                    if os.path.exists(dssp_path) and parse_dssp_check(dssp_path, seq):
                        return os.path.basename(dssp_path)
            print(f"Uid: {uid} not found in RCSB database.")


def get_uid_from_dbid(dbid):
    """
    Use HTML page of DrugBank ID (dbid) to get uid
    """
    try:
        html = requests.get(f"https://go.drugbank.com/drugs/{dbid}").content.decode()
        m = re.search('(?<=uniprot/)(.*)(?=">)', html)
        uid = html[m.start() : m.start() + 6]
        return uid
    except Exception:
        return


def parse_dssp_check(dssp_path, seq):
    dssp = parseDSSP(dssp_path)
    try:
        dssp.parse()
        if "".join(dssp.aa) == seq:
            return True
        else:
            return "".join(dssp.aa)
    except Exception:
        return None


if __name__ == "__main__":
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
    # Change the following drug smiles due to incompatibiliy with rdkit
    df["structure"].replace("[H][N]([H])([H])[Pt](Cl)(Cl)[N]([H])([H])[H]", "[NH3+]-[Pt-2](Cl)(Cl)[NH3+]", inplace=True)  ##DB000515
    df["structure"].replace("[H][N]([H])([H])[Pt]1(OC(=O)C2(CCC2)C(=O)O1)[N]([H])([H])[H]", "C1CC2(C1)C(=O)O[Pt-2]([NH3+])([NH3+])OC2=O", inplace=True)  # DB000958
    df["structure"].replace("[H][N]1([H])[C@@H]2CCCC[C@H]2[N]([H])([H])[Pt]11OC(=O)C(=O)O1", "O1C(=O)C(=O)O[Pt-2]12[NH2+]C0CCCCC0[NH2+]2", inplace=True)  # DB000526
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

    count_change = 0
    for uid in tqdm(uniprot_ids):
        r, sdict = uniprot_to_dssp(uid, df.loc[df["uniprotkb-id"] == uid]["aa-seq"].unique()[0], try_pdb=False)  # PDB sequence mostly doesn't match with uniprot sequence (full length)
        if isinstance(sdict, str):
            df.replace(df.loc[df["uniprotkb-id"] == uid]["aa-seq"].unique()[0], r, inplace=True)
            count_change += 1
        mapping.append([None, uid, r])

    if count_change >= 1:
        print(f"DSSP didn't match with aa-seq {count_change} times. Changed the aa seq with that in dssp.")

    # Commented - Fetches non-matching pdb
    # for dbid in tqdm(db_ids):
    #     uid = get_uid_from_dbid(dbid)
    #     mapping.append([dbid, uid, uniprot_to_dssp(uid)])

    dssp_df = pd.DataFrame(mapping, columns=["drugbank-id", "uniprotkb-id", "path"])
    dssp_df.to_csv(os.path.join(dirpath, "../data/dssp_mapping.csv"), index=False)
    no_path_df = dssp_df.loc[dssp_df["path"].isna()]
    df_missing_pdb_act = df["uniprotkb-id"].isin(no_path_df["uniprotkb-id"].unique())
    print(f"PDB not found for {len(no_path_df)} uniprot ids, corresponding to {df_missing_pdb_act.sum()} rows.")
    df = df.loc[~df_missing_pdb_act]

    # drugbank-id has isoforms. Mark them. Uniprot doesn't.
    df_ = df.drop_duplicates(["drugbank-id", "structure"])[["drugbank-id", "structure"]]
    sdict = {dbid: df_.loc[df["drugbank-id"] == dbid]["structure"].unique() for dbid in df_["drugbank-id"].unique()}
    for dbid in sdict:
        if len(sdict[dbid]) > 1:
            df_ = df.loc[df["drugbank-id"] == dbid]
            for i, aa in enumerate(sdict[dbid], 1):
                idxs = df_.loc[df_["structure"] == aa].index
                df.iloc[idxs, 0] += "-" + str(i)

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

    df.to_csv(os.path.join(dirpath, "../data/drug-target-moa-mod.csv"), index=False)
