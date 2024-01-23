import pandas as pd
import argparse
import os
import numpy as np


def train_test_split(df, train_size, random_state=42):
    np.random.set_state(random_state)
    i = int(train_size * df.shape[0])
    o = np.random.permutation(df.shape[0])
    df_train, df_test = np.split(np.take(df, o, axis=0), [i])
    return df_train, df_test


def final_moa_data(df_path, drug_encoding_map_file, target_encoding_map_file, dssp_map_file, output_dir="data/", only_smiles=True, train_split=1.0, random_state=42):
    """
    Convert the MOA files and data into one csv such that it will have the following rows:
    DrugID (str), Drug (str), DrugEncodingFile (str), TargetID (str), Target (str: Optional), TargetEncodingFile (str), TargetDSSPFile (str), Y (int/float)
    """

    df = pd.read_csv(df_path)
    if only_smiles:
        df = df.loc[df["type"] == "SMILES"]
    dssp_df = pd.read_csv(dssp_map_file)
    drug_enc = pd.read_csv(drug_encoding_map_file)
    target_enc = pd.read_csv(target_encoding_map_file)

    dssp_df.rename({"path": "TargetDSSPFile"}, axis=1, inplace=True)
    df_ = pd.merge(df, dssp_df[["uniprotkb-id", "TargetDSSPFile"]], "left", "uniprotkb-id")
    df_.drop_duplicates(inplace=True)

    drug_enc.rename({"ID": "drugbank-id", "filename": "DrugEncodingFile"}, axis=1, inplace=True)
    df_ = pd.merge(df_, drug_enc[["drugbank-id", "DrugEncodingFile"]], "left", "drugbank-id")
    df_.drop_duplicates(inplace=True)

    target_enc.rename({"ID": "uniprotkb-id", "filename": "TargetEncodingFile"}, axis=1, inplace=True)
    df_ = pd.merge(df_, target_enc[["uniprotkb-id", "TargetEncodingFile"]], "left", "uniprotkb-id")
    df_.drop_duplicates(inplace=True)

    df_.rename({"drugbank-id": "DrugID", "uniprotkb-id": "TargetID", "structure": "Drug", "aa-seq": "Target", "output": "Y"}, axis=1, inplace=True)
    df_ = df_[["DrugID", "Drug", "DrugEncodingFile", "TargetID", "Target", "TargetEncodingFile", "TargetDSSPFile", "Y"]]
    df_.drop_duplicates(inplace=True)
    df_.reset_index(drop=True, inplace=True)
    df_.to_csv(os.path.join(output_dir, "drug-target-moa-final.csv"), index=False)

    train_df, test_df = train_test_split(df_, train_size=max(min(train_split, 1), 0), random_state=random_state)
    train_df.to_csv(os.path.join(output_dir, "drug-target-moa-final-train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "drug-target-moa-final-test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-path", type=str, required=True)
    parser.add_argument("--drug-encoding-map-file", type=str, required=True)
    parser.add_argument("--target-encoding-map-file", type=str, required=True)
    parser.add_argument("--dssp-map-file", type=str, required=True)
    parser.add_argument("--only-smiles", action="store_true")
    parser.add_argument("--output-dir", type=str, default="data/")
    parser.add_argument("--train-split", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    final_moa_data(**args.__dict__)
