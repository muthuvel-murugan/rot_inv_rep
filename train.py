import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from sklearn.metrics import classification_report, confusion_matrix

from FCNN import *

if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # ---------------- Training Setup ----------------
    W_in = np.load("W_784x150_mnist.npy")
    cnt = np.load("cnt_784x150_mnist.npy")  #[:10]
    cnt = np.int32(cnt/2)

    c_out = [ [14, 14, 12, 12, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
             [14, 14, 12, 12, 10, 10],
             [14, 14, 12, 12]
           ]
    
    in_types = []
    for i, count in enumerate(cnt):
        in_types.extend([i] * count)
    thresholds = [15, 6, 4]  # progressive drop thresholds
    num_classes = 10
  
    X_tr = {}
    X_ts = {}
    y_tr = {}
    y_ts = {}

    with np.load("ds_mnist/mnist_org_train_784.npz") as ds:
        X = ds["data"]
        y = ds["labels"]
        X_proj = torch.tensor(X.dot(W_in)).to(device)
        X_tr["St"] = (X_proj[:, ::2] + 1j * X_proj[:, 1::2]).to(torch.cfloat)
        y_tr["St"] = torch.tensor(y).to(torch.long).to(device)
   
    with np.load("ds_mnist/mnist_org_test_784.npz") as ds:
        X = ds["data"]
        y = ds["labels"]
        X_proj = torch.tensor(X.dot(W_in)).to(device)
        X_ts["St"] = (X_proj[:, ::2] + 1j * X_proj[:, 1::2]).to(torch.cfloat)
        y_ts["St"] = torch.tensor(y).to(torch.long).to(device)
 
    with np.load("ds_mnist/mnist_train.npz") as ds:
        X = ds["data"]
        y = ds["labels"]
        X_proj = torch.tensor(X.dot(W_in)).to(device)
        X_tr["Rot"] = (X_proj[:, ::2] + 1j * X_proj[:, 1::2]).to(torch.cfloat)
        y_tr["Rot"] = torch.tensor(y).to(torch.long).to(device)
   
    with np.load("ds_mnist/mnist_test.npz") as ds:
        X = ds["data"]
        y = ds["labels"]
        X_proj = torch.tensor(X.dot(W_in)).to(device)
        X_ts["Rot"] = (X_proj[:, ::2] + 1j * X_proj[:, 1::2]).to(torch.cfloat)
        y_ts["Rot"] = torch.tensor(y).to(torch.long).to(device)
  
    train_on = "St"     
    print(f"*********************** Model Training on {train_on} ***********************") 
    # train_on = "Rot"
    tr_samp = y_tr[train_on].shape[0] # * 95 // 100

    print("Number of training samples : ", tr_samp)

    # X_train = torch.tensor(X_tr[train_on][:tr_samp, ::2] + 1j * X_tr[train_on][:tr_samp, 1::2]).to(torch.cfloat).to(device)
    # y_train = torch.tensor(y_tr[train_on][:tr_samp]).long().to(device)
    X_train = X_tr[train_on][:tr_samp]
    y_train = y_tr[train_on][:tr_samp]
    
    # X_val = torch.tensor(X_tr[train_on][tr_samp:, ::2] + 1j * X_tr[train_on][tr_samp:, 1::2]).to(torch.cfloat).to(device)
    # y_val = torch.tensor(y_tr[train_on][tr_samp:]).long().to(device)
    # Always run val on rotated samples
    X_val = X_ts["Rot"][:1000]
    y_val = y_ts["Rot"][:1000]
    
    print("Number of X_train.shape : ", X_train.shape)
    print("Number of y_train.shape : ", y_train.shape)
    print("Number of X_val.shape : ", X_val.shape)
    print("Number of y_val.shape : ", y_val.shape)
    
    model = EndToEndClassifier(torch.tensor(in_types).to(device), thresholds_per_repeat=thresholds, num_classes=num_classes, keep_d1=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Load checkpoint if exists
    ckpt_path = "classifier_checkpoint.pt"
    start_epoch = 0
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop with progress bar
    epochs = 100
    batch_size = 512
    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        pbar = tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)
        for i in pbar:
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_err = compute_error(model, X_train, y_train, batch_size)
            val_err = compute_error(model, X_val, y_val, batch_size)
            pbar.set_postfix({"TrainErr %": f"{train_err:.1f}", "ValErr %": f"{val_err:.1f}"})
    
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict()
            }, ckpt_path)
            print(f"Checkpoint saved at epoch {epoch+1}")
    
    # ---------------- Test Loop with Statistics ----------------
    print(f"*********************** Model Trained on {train_on} ***********************") 

    # ------------ Test on Straight train sets ----------------
    model.eval()
    with torch.no_grad():
        preds, labels = batched_test(model, X_tr["St"], y_tr["St"], batch_size=1024) #, device="cuda")
    overall_acc = (preds == labels).float().mean().item() * 100
    overall_err = 100 - overall_acc
    print(f"Tested on Tr - Straight ===>  Accuracy: {overall_acc:.2f}%     Error: {overall_err:.2f}%")
    
    # ------------ Test on Straight test sets ----------------
    model.eval()
    with torch.no_grad():
        preds, labels = batched_test(model, X_ts["St"], y_ts["St"], batch_size=1024) #, device="cuda")
    overall_acc = (preds == labels).float().mean().item() * 100
    overall_err = 100 - overall_acc
    print(f"Tested on Ts - Straight ===>  Accuracy: {overall_acc:.2f}%     Error: {overall_err:.2f}%")
 
    # ------------ Test on Rotated train sets ----------------
    model.eval()
    with torch.no_grad():
        preds, labels = batched_test(model, X_tr["Rot"], y_tr["Rot"], batch_size=1024) #, device="cuda")
    overall_acc = (preds == labels).float().mean().item() * 100
    overall_err = 100 - overall_acc
    print(f"Tested on Tr - Rotated  ===>  Accuracy: {overall_acc:.2f}%     Error: {overall_err:.2f}%")
    
    # ------------ Test on Rotated test sets ----------------
    model.eval()
    with torch.no_grad():
        preds, labels = batched_test(model, X_ts["Rot"], y_ts["Rot"], batch_size=1024) #, device="cuda")
    overall_acc = (preds == labels).float().mean().item() * 100
    overall_err = 100 - overall_acc
    print(f"Tested on Ts - Rotated  ===>  Accuracy: {overall_acc:.2f}%     Error: {overall_err:.2f}%")
    
    # # Per-class stats
    # print("\nClassification Report:")
    # print(classification_report(y_test.cpu(), preds.cpu(), digits=3))
    # 
    # # Confusion matrix
    # print("Confusion Matrix:")
    # print(confusion_matrix(y_test.cpu(), preds.cpu()))
    
