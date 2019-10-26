
import time

import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt

from source.dataset import cifar_dataloader
from source.model import SimpleModel
from source.model import ConvModel
from source.model import VGGLikeModel
from source.model import WideModel

# initual variables
epochs = 50
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading data and model
train, val, test = cifar_dataloader(batch_size=32)
model = WideModel()
# for gpu usage
model = model.to(device)

# optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
criterion = nn.CrossEntropyLoss()

# for plotting history of training
epoch_loss = {
    'train' : [],
    'val'   : []
}
epoch_acc  = {
    'train' : [],
    'val'   : []
}

# training phase
for epoch in range(1, epochs+1):
    print('EPOCH {} / {}'.format(epoch, epochs))
    epoch_start = time.time()
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train(True)
            dataset = train
        else:
            model.train(False)
            dataset = val
        loss    = 0
        correct = 0
        batch_count = 0
        for index, (data, target) in enumerate(dataset, 1):
            # data to gpu
            data = data.to(device)
            target = target.to(device)

            # forward
            output = model(data)
            # loss calculation
            batch_loss = criterion(output, target)

            if phase == 'train':
                # optimization
                # only on training data
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            # for calculating epoch accuracy and loss
            loss += batch_loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            batch_count += 1
        # epoch accuracy and loss calculation
        epoch_loss[phase].append(loss / batch_count)
        epoch_acc[phase].append(100 * correct / len(dataset.dataset))

    epoch_time = time.time() - epoch_start
    # verbose
    print('sec {:.2f}[s]'.format(epoch_time),                    end='\t')
    print('Train loss : {:.5f}'.format(epoch_loss['train'][-1]), end='\t')
    print( 'Train Acc : {:.5f}'.format(epoch_acc['train'][-1]),  end='\t')
    print(  'Val loss : {:.5f}'.format(epoch_loss['val'][-1]),   end='\t')
    print(   'Val Acc : {:.5f}'.format(epoch_acc['val'][-1]))

# evaluate
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

# plotting
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