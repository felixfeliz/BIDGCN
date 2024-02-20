from math import floor
import torch.utils.data
import os
from torch_geometric.loader import DataLoader

from dataset import TensorProductPolynomialData
from architecture import ParameterizationNet, lossfunction

torch.manual_seed(3)

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Using " + str(device))

torch.set_default_dtype(torch.float64)

surfaceDegree = 2
fittingDegree = 2
num_epochs = 10000



if not os.path.isdir("logs"):
    os.makedirs("logs")
if not os.path.isdir("models"):
    os.makedirs("models")

name = "paramnet"

log = open(f"logs/log{name}.csv", "w")
log.write("trainingerror, validationerror, trainingparametererror, validationparametererror\n")


dataset = TensorProductPolynomialData("./data/", length=100000, n=1000, d=surfaceDegree)

train_set, val_set = torch.utils.data.random_split(dataset,
                                                   [int(floor(len(dataset) * 0.8)), int(floor(len(dataset) * 0.2))])

trainLoader = DataLoader(train_set, batch_size=1, shuffle=True)
valLoader = DataLoader(val_set, batch_size=1, shuffle=False)


net = ParameterizationNet()
net.to(device)

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)


def train():
    net.train()
    running_loss = 0.0
    parameter_loss = 0.0
    for idx, batch in enumerate(trainLoader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = net(batch)

        loss = lossfunction(out, batch.interior, batch.boundary, degree=fittingDegree)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step(running_loss)
    return running_loss / len(trainLoader.dataset), parameter_loss / len(trainLoader.dataset)

def test():
    net.eval()
    running_loss = 0.0
    parameter_loss = 0.0
    with torch.no_grad():
        for idx, batch in enumerate(valLoader):
            batch = batch.to(device)
            out = net(batch)
            running_loss += lossfunction(out, batch.interior, batch.boundary, degree=fittingDegree)
    return running_loss / len(valLoader.dataset), parameter_loss / len(valLoader.dataset)



print(f"Start training for {num_epochs} epochs.\n"
      f"Number of learnable parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad)}")
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    trainloss, trainparameterloss = train()
    validationloss, validationparameterloss = test()
    log.write(f"{trainloss}, {validationloss}, {trainparameterloss}, {validationparameterloss}\n")
    log.flush()

    torch.save(net.state_dict(), f"models/{name}Epoch{epoch}.pt")

    print(f"Training loss after epoch {epoch} is {trainloss}. Parameter loss: {trainparameterloss}\n"
          f"Validation loss after epoch {epoch} is {validationloss}. Parameter loss: {validationparameterloss}")

log.close()
