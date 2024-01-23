import os
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
from model import DTI_MOA_Model
from data_utils import MOADataset, collate_fn
from tqdm import tqdm


def main(args):
    device = torch.device(args.device)
    train_dataset = MOADataset(pd.read_csv(args.train_df_path), args.smiles_encoding_root, args.prot_encoding_root, args.dssp_root, args.num_classes)
    test_dataset = MOADataset(pd.read_csv(args.test_df_path), args.smiles_encoding_root, args.prot_encoding_root, args.dssp_root, args.num_classes)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = DTI_MOA_Model(args.num_classes, args.drug_enc_dim, args.drug_graph_dim, args.drug_avg_len, args.target_enc_dim, args.target_dssp_dim, args.target_avg_len)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    metrics = [torchmetrics.classification.MulticlassAccuracy(args.num_classes).to(device), torchmetrics.classification.MulticlassRecall(args.num_classes).to(device), torchmetrics.classification.MulticlassPrecision(args.num_classes).to(device), torchmetrics.classification.MulticlassF1Score(args.num_classes).to(device)]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    max_acc = 0
    max_tries = 10
    num_tries = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        print(f"Epoch: {epoch}/{args.epochs}")
        total_loss = 0
        for [target_enc, target_dssp, drug_enc, drug_graph], labels in tqdm(train_loader):
            optimizer.zero_grad()
            target_enc = [x.to(device) for x in target_enc]
            target_dssp = [x.to(device) for x in target_dssp]
            drug_enc = [x.to(device) for x in drug_enc]
            drug_graph = [x.to(device) for x in drug_graph]
            labels = labels.to(device)
            try:
                outputs = model(drug_enc, drug_graph, target_enc, target_dssp)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                for metric in metrics:
                    metric.update(torch.softmax(outputs, -1), labels)
            except Exception as e:
                print(e)

        print("Train Loss:", total_loss)
        for metric in metrics:
            print(metric, metric.compute().item())
            metric.reset()

        print()
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for [target_enc, target_dssp, drug_enc, drug_graph], labels in tqdm(test_loader):
                target_enc = [x.to(device) for x in target_enc]
                target_dssp = [x.to(device) for x in target_dssp]
                drug_enc = [x.to(device) for x in drug_enc]
                drug_graph = [x.to(device) for x in drug_graph]
                labels = labels.to(device)
                try:
                    outputs = model(drug_enc, drug_graph, target_enc, target_dssp)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item()
                    for metric in metrics:
                        metric.update(torch.softmax(outputs, -1), labels)
                except Exception as e:
                    print(e)

        print("Test Loss:", total_loss)
        for metric in metrics:
            c = metric.compute().item()
            print(metric, c)
            if "Acc" in metric.__str__():
                if c > max_acc:
                    print("New max accuracy!")
                    max_acc = c
                    num_tries = 0
                    torch.save(model, "model.pt")
                    torch.save(model.state_dict(), "model_state_dict.pt")
                else:
                    num_tries += 1
            metric.reset()
        print()

        scheduler.step()
        if num_tries == max_tries:
            print("\nExiting Training Loop")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-df-path", type=str, required=True, help="Path to final moa csv")
    parser.add_argument("--test-df-path", type=str, required=True, help="Path to final moa csv")
    parser.add_argument("--smiles-encoding-root", type=str, required=True)
    parser.add_argument("--prot-encoding-root", type=str, required=True)
    parser.add_argument("--dssp-root", type=str, required=True)
    parser.add_argument("--drug-enc-dim", type=int, required=True)
    parser.add_argument("--drug-graph-dim", type=int, required=True)
    parser.add_argument("--drug-avg-len", type=int, default=64)
    parser.add_argument("--target-enc-dim", type=int, required=True)
    parser.add_argument("--target-dssp-dim", type=int, required=True)
    parser.add_argument("--target-avg-len", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    main(parser.parse_args())