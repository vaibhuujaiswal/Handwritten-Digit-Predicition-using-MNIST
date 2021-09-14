### YOUR CODE HERE
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

#loading data and shuffling :

mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

W1 = torch.randn(784,500)/np.sqrt(784)
W1.requires_grad_()
b1 = torch.zeros(500, requires_grad = True)

W2 = torch.randn(500,10)/np.sqrt(500)
W2.requires_grad_()
b2 = torch.zeros(10, requires_grad= True)


optimizer = torch.optim.SGD([W2,b2],lr = 0.05)

epoch = 1
for i in range(0,epoch):
    for images,labels in tqdm(train_loader):
        #zero out optimizer
        optimizer.zero_grad()
        
        #forward pass
        x = images.view(-1,28*28)
        #input image
        z = torch.matmul(x,W1) + b1
#         cross_entropy = F.cross_entropy(z, labels)
#         cross_entropy.backward()
#         optimizer.step()
        A = F.relu(z) #y is 100 X 500 ----> we have to convert this in 100 X 10 #activation function rectified linear unit.
        y = torch.matmul(A, W2) + b2

        #backward pass
        cross_entropy = F.cross_entropy(y, labels)
        cross_entropy.backward()
        optimizer.step()
        
    
#testing
correct = 0
total = len(mnist_test) #which is 10,000

with torch.no_grad(): #don't need to calculate gradient
    for images, labels in tqdm(test_loader) :
        x = images.view(-1,28*28)
        z = torch.matmul(x,W1) + b1
        A = F.relu(z)
        y = torch.matmul(A,W2) + b2
        
        predictions = torch.argmax(y,dim = 1) #calculating the max probability
        correct += torch.sum((predictions == labels).float())

print('Test accuracy: {}'.format(correct/total))
print(correct)
print(total)

# Make sure to print out your accuracy on the test set at the end.