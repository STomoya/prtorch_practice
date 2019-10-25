
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

from source.dataset import cifar_dataloader
from source.model import SimpleModel

epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train, val, test = cifar_dataloader(batch_size=50)
model = SimpleModel()
model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

epoch_loss = {
    'train' : [],
    'val'   : []
}
epoch_acc  = {
    'train' : [],
    'val'   : []
}

model.train()
for epoch in range(1, epochs+1):
    print('EPOCH {} / {}'.format(epoch, epochs))
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)
            dataset = train
        else:
            model.train(False)
            dataset = val
        loss     = 0
        correct = 0
        for index, (data, target) in enumerate(dataset, 1):
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            batch_loss = criterion(output, target)

            if phase == 'train':
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            loss += batch_loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
        epoch_loss[phase].append(loss/(len(dataset.dataset)/dataset.batch_size))
        epoch_acc[phase].append(100*correct/len(dataset.dataset))

    print('Train loss : {:.5f}'.format(epoch_loss['train'][-1]), end='\t')
    print( 'Train Acc : {:.5f}'.format(epoch_acc['train'][-1]),  end='\t')
    print(  'Val loss : {:.5f}'.format(epoch_loss['val'][-1]),   end='\t')
    print(   'Val Acc : {:.5f}'.format(epoch_acc['val'][-1]))

correct = 0
total = 0
model.eval()
with torch.no_grad():
    for (data, target) in test:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = output.max(1)
        correct += (predicted == target).sum().item()
print('test accuracy : {:.5f}'.format(100*correct/len(test.dataset)))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_loss['train'])
plt.plot(epoch_loss['val'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(epoch_acc['train'])
plt.plot(epoch_acc['val'])
plt.title('model accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'validation'], loc='upper left')

plt.show()