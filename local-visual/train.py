import torch.nn as nn
import torch
from test import eval

def train_AE(model, epoch, train_loader, test_dataset, optimizer, labels, model_name, device, t):
    mse = nn.MSELoss().to(device)
    batch_len = len(train_loader)
    best_auc = 0.0 
    for i in range(epoch):
        model.train()
        total_loss = 0.0
        for pst, prst, ftr in train_loader:
            optimizer.zero_grad()
            feature = pst.shape[-1]
            pst = pst.to(device).reshape(-1, feature)
            prst = prst.to(device).reshape(-1, feature)
            ftr = ftr.to(device).reshape(-1, feature)
            hat_pst, hat_prst, hat_ftr = model(prst)
            loss_pst = mse(hat_pst, pst)
            loss_prst = mse(hat_prst, prst)
            loss_ftr = mse(hat_ftr, ftr)
            loss = loss_pst + loss_prst + loss_ftr
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            model.eval()
            auc, _ = eval(model, test_dataset, labels, device=device, t=t)
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), f'./results/models/{model_name}.pth')
            
            print(f'cur_epoch:{i+1}, loss:{total_loss/batch_len}, auc:{auc}')
    return best_auc