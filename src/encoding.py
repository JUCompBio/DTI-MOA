import sys
import os
import argparse
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
import preprocessing_utils

dirpath = os.path.dirname(__file__)


def encode_and_save(model, seq, seq_id, root, prot=True):
    file_name = seq_id
    if prot:
        file_name += "_" + seq[:200]
        seq = " ".join(list(prot))

    torch.save(model(seq), os.path.join(root, f"{file_name}.pt"))
    return f"{file_name}.pt"


# Encode the targets and drugs
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-path", type=str, required=True, help="Path to DTI-MOA csv df")
    parser.add_argument("--smiles", action="store_true", help="Encode SMILES")
    parser.add_argument("--aa", action="store_true", help="Encode Proteins")
    parser.add_argument("--smiles-model", type=str, help="Hugging-face model to encode SMILES")
    parser.add_argument("--aa-model", type=str, help="Hugging-face model to encode Proteins")
    parser.add_argument("--out-df-filename", type=str, default="encoder_mapping.csv", help="Save id-file mapping in a dataframe")
    args = parser.parse_args(args)

    if args.smiles and (args.smiles_model is None):
        parser.error("SMILES model required for encoding SMILES")

    if args.aa and (args.aa_model is None):
        parser.error("Protein model required for encoding Proteins")

    device = 0 if torch.cuda.is_available() else -1
    df = pd.read_csv(args.df_path)
    mapping = []  # columns - id (protein/drug), sequence, file path
    prot_enc_dir = os.path.join(dirpath, "../data/prot_enc", args.aa_model)
    os.makedirs(prot_enc_dir, exist_ok=True)
    smiles_enc_dir = os.path.join(dirpath, "../data/drug_enc", args.smiles_model)
    os.makedirs(smiles_enc_dir, exist_ok=True)

    if args.aa:
        prot_model = preprocessing_utils.MyFeatureExtractionPipeline(task="feature-extraction", model=AutoModel.from_pretrained(args.aa_model), tokenizer=AutoTokenizer.from_pretrained(args.aa_model), return_tensors=True, device=device)

        df_ = df.drop_duplicates(subset=["uniprotkb-id", "aa-seq"])
        for _, row in tqdm(df_.iterrows(), total=len(df_)):
            f = encode_and_save(prot_model, row["aa-seq"], row["uniprotkb-id"], prot_enc_dir)
            mapping.append([row["uniprotkb-id"], row["aa-seq"], f])

        df_ = df.loc[df["type"] == "polypeptide"].drop_duplicates(subset=["drugbank-id", "structure"])
        for _, row in tqdm(df_.iterrows(), total=len(df_)):
            f = encode_and_save(prot_model, row["structure"], row["drugbank-id"], prot_enc_dir)
            mapping.append([row["drugbank-id"], row["structure"], f])

    del prot_model

    if args.smiles:
        smiles_model = preprocessing_utils.MyFeatureExtractionPipeline(task="feature-extraction", model=AutoModel.from_pretrained(args.smiles_model), tokenizer=AutoTokenizer.from_pretrained(args.smiles_model), return_tensors=True, device=device)
        df_ = df.loc[df["type"] == "SMILES"].drop_duplicates(subset=["drugbank-id", "structure"])
        for _, row in tqdm(df_.iterrows(), total=len(df_)):
            f = encode_and_save(smiles_model, row["structure"], row["drugbank-id"], smiles_enc_dir, False)
            mapping.append([row["drugbank-id"], row["structure"], f])

    pd.DataFrame(mapping, columns=["ID", "Seq", "filename"]).to_csv(os.path.join(dirpath, "../data", args.out_df_filename), index=False)


if __name__ == "__main__":
    main(sys.argv[1])
