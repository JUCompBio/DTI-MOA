import sys
import os
import argparse
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline

dirpath = os.path.dirname(__file__)


prot_model_dict = {
    "bert": {"model": BertModel, "model_args": ["Rostlab/prot_bert"], "model_kwargs": {}, "tokenizer": BertTokenizer, "tokenizer_args": ["Rostlab/prot_bert"], "tokenizer_kwargs": {}},
    "t5": {"model": T5EncoderModel, "model_args": ["Rostlab/prot_t5_xl_uniref50"], "model_kwargs": {}, "tokenizer": T5Tokenizer, "tokenizer_args": ["Rostlab/prot_t5_xl_uniref50"], "tokenizer_kwargs": {"do_lower_case": False}},
    "esm2": {"model": AutoModel, "model_args": ["facebook/esm2_t33_650M_UR50D"], "model_kwargs": {}, "tokenizer": AutoTokenizer, "tokenizer_args": ["facebook/esm2_t33_650M_UR50D"], "tokenizer_kwargs": {}},
}


class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def preprocess(self, inputs):
        model_inputs = self.tokenizer(inputs, add_special_tokens=False, padding=False, return_tensors=self.framework)
        return model_inputs


def encode_prot_and_save(pipeline, seq, filename, root):
    seq = " ".join(list(seq))
    output = pipeline(seq)
    torch.save(output.detach().cpu(), os.path.join(root, filename + ".pt"))


# Encode the targets and drugs
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-path", type=str, required=True, help="Path to DTI-MOA csv df")
    parser.add_argument("--smiles", action="store_true", help="Encode SMILES")
    parser.add_argument("--aa", action="store_true", help="Encode Proteins")
    parser.add_argument("--smiles-model", type=str, help="Hugging-face model to encode SMILES")
    parser.add_argument("--aa-model", type=str, choices=["bert", "t5", "esm2"], help="Hugging-face model to encode Proteins")
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
        config = prot_model_dict[args.aa_model]
        prot_pipeline = MyFeatureExtractionPipeline(task="feature-extraction", model=config["model"](*config["model_args"], **config["model_kwargs"]), tokenizer=config["tokenizer"](*config["tokenizer_args"], **config["tokenizer_kwargs"]), return_tensors=True, device=device)

        df_ = df.drop_duplicates(subset=["uniprotkb-id", "aa-seq"])
        for _, row in tqdm(df_.iterrows(), total=len(df_)):
            encode_prot_and_save(prot_pipeline, " ".join(list(row["aa-seq"])), row["uniprotkb-id"], prot_enc_dir)
            mapping.append([row["uniprotkb-id"], row["aa-seq"], row["uniprotkb-id"] + ".pt"])

        df_ = df.loc[df["type"] == "polypeptide"].drop_duplicates(subset=["drugbank-id", "structure"])
        for _, row in tqdm(df_.iterrows(), total=len(df_)):
            encode_prot_and_save(prot_pipeline, " ".join(list(row["structure"])), row["drugbank-id"], prot_enc_dir)
            mapping.append([row["drugbank-id"], row["structure"], row["drugbank-id"] + ".pt"])

        del prot_model

    if args.smiles:
        # use MolCLR
        pass

    pd.DataFrame(mapping, columns=["ID", "Seq", "filename"]).to_csv(os.path.join(dirpath, "../data", args.out_df_filename), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
