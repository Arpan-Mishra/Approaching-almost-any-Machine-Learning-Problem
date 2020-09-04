import torch
import torch.nn as nn
from tqdm import tqdm

def train(data_loader, model, optimizer, device):

    model.train()

    for data in data_loader:
        inputs = data['image']
        targets = data['targets']

        inputs = inputs.to(device, dtype = torch.float)
        targets = targets.to(device, dtype = torch.float)

        optimizer.zero_grad()
        output = model(inputs)

        loss = nn.BCEWithLogitsLoss()(output, targets.view(-1,1))
        
        loss.backward()
        optimizer.step()

def evaluate(data_loader, model, device):
    model.eval()

    final_targets = []
    final_outputs = []

    with torch.no_grad():
        for data in data_loader:
            inputs = data['image']
            targets = data['targets']

            inputs = inputs.to(device, dtype = torch.float)
            targets = targets.to(device, dtype = torch.float)

            output = model(inputs)
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(output)

    return final_outputs, final_targets


