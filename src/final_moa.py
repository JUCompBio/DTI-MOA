import pandas as pd
import argparse


def final_moa_data(df_path, drug_encoding_map_file, target_encoding_map_file, dssp_map_file, only_smiles=True):
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

    drug_enc.rename({"ID": "drugbank-id", "filename": "DrugEncodingFile"}, axis=1, inplace=True)
    df_ = pd.merge(df_, drug_enc[["drugbank-id", "DrugEncodingFile"]], "left", "drugbank-id")

    target_enc.rename({"ID": "uniprotkb-id", "filename": "TargetEncodingFile"}, axis=1, inplace=True)
    df_ = pd.merge(df_, target_enc[["uniprotkb-id", "TargetEncodingFile"]], "left", "uniprotkb-id")

    df_.rename({"drugbank-id": "DrugID", "uniprotkb-id": "TargetID", "structure": "Drug", "aa-seq": "Target", "output": "Y"}, axis=1, inplace=True)
    df_ = df_[["DrugID", "Drug", "DrugEncodingFile", "TargetID", "Target", "TargetEncodingFile", "TargetDSSPFile", "Y"]]

    df_.to_csv("drug-target-moa-final.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-path", type=str, required=True)
    parser.add_argument("--drug-encoding-map-file", type=str, required=True)
    parser.add_argument("--target-encoding-map-file", type=str, required=True)
    parser.add_argument("--dssp-map-file", type=str, required=True)
    parser.add_argument("--only-smiles", action="store_true")
    args = parser.parse_args()
    final_moa_data(**args.__dict__)
