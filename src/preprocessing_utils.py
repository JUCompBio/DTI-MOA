import re
import os
import json
import requests
import urllib
import random
import numpy as np
from rdkit import Chem
import gzip
import shutil
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
from utils import parse_dssp_check


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
            if parse_dssp_check(dssp_path, seq):
                return dssp_path
            else:
                print("DSSP for uid: {uid} could not be generated due to non-matching aa-seq.")
        else:
            print("DSSP for uid: {uid} could not be generated.")
    else:
        print(f"Alphafold pdb for uid: {uid} not found.")
        if try_pdb:
            print("Trying RCSB database...")
            for pdb_id in get_pdb_from_uid(uid):
                pdb_path = download_pdb_pdb_file(pdb_id, os.path.join(dirpath, "../data/pdb/"))
                if pdb_path:
                    dssp_path = perform_dssp(os.path.join(dirpath, "../data/pdb/"), os.path.join(dirpath, "../data/dssp/"), pdb_id)
                    if os.path.exists(dssp_path) and parse_dssp_check(dssp_path, seq):
                        return dssp_path
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


class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def preprocess(self, inputs):
        return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, padding=True)
        return model_inputs
