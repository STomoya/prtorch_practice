
import torch
import torch.optim as optim
import torch.nn as nn

from source.dataset import cifar_dataloader
from source.model import SimpleModel

epochs = 10
log_delay = 200

train, test = cifar_dataloader(batch_size=50, validation=False)
model = SimpleModel()

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(1, epochs+1):
    print('EPOCH {} / {}'.format(epoch, epochs))
    total_train=0
    correct_train=0
    for index, (data, target) in enumerate(train, 1):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total_train += target.size(0)
        correct_train += (predicted == target).sum().item()
        if index % log_delay == 0:
            print('[{}/{}]\tLoss : {:.5f}\tAccuacy : {:.5f}'.format(index * len(data), len(train.dataset), loss.data, 100*correct_train/total_train))
    
correct = 0
total = 0
with torch.no_grad():
    for (data, target) in test:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
print('test accuracy : {:.5f}'.format(100*correct/total))