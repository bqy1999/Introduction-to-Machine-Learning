import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torch import  nn
from torch.autograd import Variable
from torch import  optim
from torchvision import transforms
from matplotlib import cm
import matplotlib.pyplot as plt

try: from sklearn.manifold import TSNE; HAS_SK = True

except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

plt.ion()

# hyper parameters
LR = 0.01
EPOCH = 2
nums_epoch = 20
#  nums_epoch = 1

# define CNN
                nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16,32,kernel_size=3),# 32, 24, 24
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2)) # 32, 12,12     (24-2) /2 +1
        self.layer3 = nn.Sequential(
                nn.Conv2d(32,64,kernel_size=3), # 64,10,10
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(
                nn.Conv2d(64,128,kernel_size=3),  # 128,8,8
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2,stride=2))  # 128, 4,4
        self.fc = nn.Sequential(
                nn.Linear(128 * 4 * 4,1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024,128),
                nn.ReLU(inplace=True),
                nn.Linear(128,10))
        return

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        output = self.fc(x)
        return output, x
    #  return output

# download mnist dataset
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

# preprocess
data_tf = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5])]
        )

train_set = mnist.MNIST('./data',train=True,transform=data_tf,download=True)
test_set = mnist.MNIST('./data',train=False,transform=data_tf,download=True)

train_data = DataLoader(train_set,batch_size=64,shuffle=True, num_workers=2)
test_data = DataLoader(test_set,batch_size=128,shuffle=False)

Net = CNN()
print(Net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(Net.parameters(), lr=1e-1)
#  optimizer = optim.Adam(Net.parameters(), lr=1e-1, betas=(0.9, 0.99))


# start training
losses =[]
losses_his =[]
acces = []
eval_losses = []
eval_acces = []

for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    Net = Net.train()
    #  train
    for img , label in train_data:
        #img = img.reshape(img.size(0),-1)
        img = Variable(img)
        label = Variable(label)

        # forward propagation
        out, last_layer = Net(img)
        loss = criterion(out,label)
        # backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        train_loss += loss.item()
        # calculate accuarcy
        _ , pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
        train_loss2 = 0
#          for img2 , label2 in test_data:
            #  img2 = Variable(img2)
            #  label2 = Variable(label2)
            #  out2, last_layer2 = Net(img2)
            #
            #  #  loss function
            #  loss2 = criterion(out2, label2)
            #  train_loss2 += loss2.item()
        #  losses_his.append(train_loss2/ len(test_data))
        #  print("end")

    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0

    #  test
    for img , label in test_data:
        #img = img.reshape(img.size(0),-1)
        img = Variable(img)
        label = Variable(label)
        out, last_layer = Net(img)

        #  loss function
        loss = criterion(out,label)

        # record loss and accuarcy
        eval_loss += loss.item()
        _ , pred = out.max(1)
        num_correct = (pred==label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc

        if HAS_SK:
            # Visualization of trained flatten layer (T-SNE)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            plot_only = 500
            low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
            labels = label.numpy()[:plot_only]
            plot_with_labels(low_dim_embs, labels)

    #  record loss and accuarcy
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    #  output
    print('Epoch:{}'.format(epoch+1))
    print('Test Loss:{}, Test Accuracy:{}'.format((eval_loss/len(test_data)), (eval_acc/len(test_data))))
    continue

#  print('hello')
#  print(losses_his)

#  print(losses)
#  for l_his in enumerate(losses):
    #  print('hello')
#  plt.plot(losses, label="Training Loss")
#  plt.plot(eval_losses, label="Validation Loss")
#  plt.legend(loc='best')
#  plt.xlabel('Epochs')
#  plt.ylabel('Loss')
#  plt.ylim((0, 2))
#  plt.show()
plt.ioff()

#  net_SGD = CNN()
#  net_Momentum = CNN()
#  net_RMSprop = CNN()
#  net_Adam = CNN()
#  nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]
#
#  #  optimizer = optim.SGD(Net.parameters(),1e-1)
#  opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
#  opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
#  opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
#  opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
#  optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]
#
#  loss_func = nn.CrossEntropyLoss()
#  losses_his = [[],[],[],[]]
#
#  for epoch in range(EPOCH):
#      print(epoch)
#      for step, (batch_x, batch_y) in enumerate(train_data):
#          b_x = Variable(batch_x)
#          b_y = Variable(batch_y)
#
#          for net, opt, l_his in zip(nets, optimizers, losses_his):
#              output = net(b_x)
#              loss = loss_func(output, b_y)
#              opt.zero_grad()
#              loss.backward()
#              opt.step()
#              l_his.append(loss.item())
#
#  labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
#  for i, l_his in enumerate(losses_his):
#      plt.plot(l_his, label=labels[i])
#
#  plt.legend(loc='best')
#  plt.xlabel('Steps')
#  plt.ylabel('Loss')
#  plt.ylim((0, 1))
#  plt.show()
