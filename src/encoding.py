import sys
import os
import argparse
import pandas as pd
import selfies as sf
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer, T5EncoderModel, T5Tokenizer
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline

dirpath = os.path.dirname(__file__)


model_dict = {
    "bert": {"model": BertModel, "model_args": ["Rostlab/prot_bert"], "model_kwargs": {"add_pooling_layer": False}, "tokenizer": BertTokenizer, "tokenizer_args": ["Rostlab/prot_bert"], "tokenizer_kwargs": {}, "data_indices": [1, -1], "type": "prot", "lib": "hf"},
    "t5": {"model": T5EncoderModel.from_pretrained, "model_args": ["Rostlab/prot_t5_xl_uniref50"], "model_kwargs": {}, "tokenizer": T5Tokenizer, "tokenizer_args": ["Rostlab/prot_t5_xl_uniref50"], "tokenizer_kwargs": {"do_lower_case": False}, "data_indices": [0, -1], "type": "prot", "lib": "hf"},
    "esm2": {"model": AutoModel, "model_args": ["facebook/esm2_t33_650M_UR50D"], "model_kwargs": {"add_pooling_layer": False}, "tokenizer": AutoTokenizer, "tokenizer_args": ["facebook/esm2_t33_650M_UR50D"], "tokenizer_kwargs": {}, "data_indices": [1, -1], "type": "prot", "lib": "hf"},
    "selformer": {"model": AutoModel, "model_args": ["HUBioDataLab/SELFormer"], "model_kwargs": {"add_pooling_layer": False}, "tokenizer": AutoTokenizer, "tokenizer_args": ["HUBioDataLab/SELFormer"], "tokenizer_kwargs": {}, "data_indices": [1, -1], "type": "smiles", "lib": "hf"},
    "chemberta": {"model": AutoModel, "model_args": ["seyonec/PubChem10M_SMILES_BPE_450k"], "model_kwargs": {"add_pooling_layer": False}, "tokenizer": AutoTokenizer, "tokenizer_args": ["seyonec/PubChem10M_SMILES_BPE_450k"], "tokenizer_kwargs": {}, "data_indices": [1, -1], "type": "smiles", "lib": "hf"},
}


class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def preprocess(self, inputs):
        model_inputs = self.tokenizer(inputs, add_special_tokens=True, padding=False, return_tensors=self.framework)
        return model_inputs


def encode_and_save(pipeline, inds, seq, filename, root):
    seq = " ".join(list(seq))
    output = pipeline(seq)
    torch.save(output.detach().cpu()[inds[0] : inds[1]], os.path.join(root, filename + ".pt"))


# Encode the targets and drugs
def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--df-path", type=str, required=True, help="Path to DTI-MOA csv df")
    parser.add_argument("--model", type=str, nargs="*", default=[], choices=list(model_dict.keys()) + ["all"], help="Hugging-face model(s) to encode Proteins/SMILES")
    parser.add_argument("--out-df-filename", type=str, default="encoder_mapping.csv", help="Save id-file mapping in a dataframe")
    args = parser.parse_args(args)
    if "all" in args.model:
        args.model = list(model_dict.keys())

    device = 0 if torch.cuda.is_available() else -1
    df = pd.read_csv(args.df_path)
    mapping = []  # columns - id (protein/drug), model_id, sequence, file path

    for model in args.model:
        config = model_dict[model]
        enc_dir = os.path.join(dirpath, f"../data/{config['type']}/", model)
        os.makedirs(enc_dir, exist_ok=True)
        if config["lib"] == "hf":
            pipeline = MyFeatureExtractionPipeline(task="feature-extraction", model=config["model"](*config["model_args"], **config["model_kwargs"]), tokenizer=config["tokenizer"](*config["tokenizer_args"], **config["tokenizer_kwargs"]), return_tensors=True, device=device)

        if config["type"] == "prot":
            df_ = df.drop_duplicates(subset=["uniprotkb-id", "aa-seq"])
            for _, row in tqdm(df_.iterrows(), total=len(df_)):
                encode_and_save(pipeline, config["model"]["data_indices"], " ".join(list(row["aa-seq"])), row["uniprotkb-id"], enc_dir)
                mapping.append([row["uniprotkb-id"], model, row["aa-seq"], row["uniprotkb-id"] + ".pt"])

            df_ = df.loc[df["type"] == "polypeptide"].drop_duplicates(subset=["drugbank-id", "structure"])
            for _, row in tqdm(df_.iterrows(), total=len(df_)):
                encode_and_save(pipeline, config["model"]["data_indices"], " ".join(list(row["structure"])), row["drugbank-id"], enc_dir)
                mapping.append([row["drugbank-id"], model, row["structure"], row["drugbank-id"] + ".pt"])

        if config["type"] == "smiles":
            df_ = df.loc[df["type"] == "SMILES"].drop_duplicates(subset=["drugbank-id", "structure"])
            if config["lib"] == "hf":
                for _, row in tqdm(df_.iterrows(), total=len(df_)):
                    encode_and_save(pipeline, config["model"]["data_indices"], sf.encoder(row["structure"]) if model == "selformer" else row["structure"], row["drugbank-id"], enc_dir)
                    mapping.append([row["drugbank-id"], model, row["structure"], row["drugbank-id"] + ".pt"])

        if config["lib"] == "hf":
            del pipeline

        # use GIN MolCLR

    pd.DataFrame(mapping, columns=["ID", "Model", "Seq", "filename"]).to_csv(os.path.join(dirpath, "../data", args.out_df_filename), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
