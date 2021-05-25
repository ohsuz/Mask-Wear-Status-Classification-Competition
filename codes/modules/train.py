import torch
import numpy as np
from .utils import use_cuda
from .config import Config as conf
from .config import HyperParameter as params

device = use_cuda()

def train_and_validate(model, criterion, optimizer, scheduler, train_loader, len_val, val_loader):
    for epoch in range(params.NUM_EPOCHS):
        model.train()
        loss_value = 0
        matches = 0
        
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % params.train_log_interval == 0:
                train_loss = loss_value / params.train_log_interval
                train_acc = matches / params.BATCH_SIZE / params.train_log_interval
                current_lr = scheduler.get_last_lr()
                print(
                    f"Epoch[{epoch}/{params.NUM_EPOCHS}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )

                loss_value = 0
                matches = 0
        
        scheduler.step()
        if validate(model, criterion, len_val, val_loader) == -1:
            break
                

def validate(model, criterion, len_val, loader):
    counter = 0
    best_val_acc = 0
    best_val_loss = np.inf
    
    with torch.no_grad():
        print("Calculating validation results...")
        model.eval()
        val_loss_items = []
        val_acc_items = []
        
        for val_batch in loader:
            inputs, labels = val_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            loss_item = criterion(outs, labels).item()
            acc_item = (labels == preds).sum().item()
            val_loss_items.append(loss_item)
            val_acc_items.append(acc_item)

        val_loss = np.sum(val_loss_items) / len(loader)
        val_acc = np.sum(val_acc_items) / len_val
    
        # Callback1: validation accuracy가 향상될수록 모델을 저장합니다.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        if val_acc > best_val_acc:
            print("New best model for val accuracy! saving the model..")
            #torch.save(model.state_dict(), f"resnet_{epoch:03}_accuracy_{val_acc:4.2%}.ckpt")
            best_val_acc = val_acc
            counter = 0
        else:
            counter += 1
        # Callback2: patience 횟수 동안 성능 향상이 없을 경우 학습을 종료시킵니다.
        if counter > params.patience:
            print("Early Stopping...")
            return -1


        print(
            f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
            f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
        )
        return 1