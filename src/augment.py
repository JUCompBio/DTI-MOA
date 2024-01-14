import sys
import argparse
import os
import pandas as pd
import random
import preprocessing_utils

seed = 42
random.seed(seed)
dirpath = os.path.dirname(__file__)


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
                new_rows.extend(preprocessing_utils.augment_aa_seq_num_and_create_rows(row, -2, 2, args.max_prot_change_ratio, num, tries))

    # Temp variable just to say the whole dataset again to augment for drugs
    new_df_ = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
    new_df_ = new_df_.loc[new_df_["action"].isin(actions_aug)].values.tolist()

    # Augment the drug (SMILES or Protein)
    for row in random.sample(new_df_, int(len(new_rows))):
        if row[7] == "SMILES":
            for i in range(random.choice([0, 1, 1, 2, 2])):
                new_row = row.copy()
                new_row[6] = preprocessing_utils.randomize_smiles(new_row[6])
                new_row[0] += f"_{i}"
                new_rows.append(new_row)
        else:
            new_rows.extend(preprocessing_utils.augment_aa_seq_num_and_create_rows(row, 6, 0, args.max_prot_change_ratio, random.choice([0, 1, 1, 2, 2]), tries))

    print("Number of rows added:", len(new_rows))
    df = pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)
    df.to_csv(os.path.join(dirpath, "../data/drug-target-moa-aug.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
