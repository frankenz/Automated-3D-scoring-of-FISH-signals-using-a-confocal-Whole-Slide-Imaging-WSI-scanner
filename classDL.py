

# Algorithm developed by: Ziv Frankenstein, Ph.D. 

### PseudoCode


import torch.utils.data as data
import torchvision
from torchvision import transforms


BATCH_SIZE = 9
TRAIN_DATA_PATH = "./data/new/train"
TEST_DATA_PATH = "./data/new/test"
TRANSFORM_IMG = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225] )
    ])

# 1. Load training and test datasets 	
train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)

trainloader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)                                                                                  
testloader = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)                                  

classes = ('Picnic', 'Ice')

# 2. Define a Convolutional Neural Network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 6, 5) 
        self.fc1 = nn.Linear(6 * 61* 61, 1024)
        self.fc2 = nn.Linear(1024, 9) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 6 * 61 * 61)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
print(net)
save_net = net


# 3. Define a Loss function and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.99)


# 4. Train the network
for epoch in range(8):  

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
torch.save({'state_dict': save_net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                   }, 'last_brain1.pth')
                       


# 5. Test the network on the test data
dataiter = iter(testloader)
images, labels = dataiter.next()

print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(3)))  # of img in test folders

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(3)))  # of img in test folders

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
    
    
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)-1):
            label = labels[i]
            print ('here',label,i)
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):  
    if class_total[i] > 0:
      print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))




