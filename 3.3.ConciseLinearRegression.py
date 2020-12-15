import torch
import numpy as np
import torch.utils.data as Data
from torch import nn
from torch.nn import init
import torch.optim as optim

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)

batch_size = 10
# 通过torch的data读取
dataset = Data.TensorDataset(features,labels)
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)
# method 1
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet,self).__init__()
        self.linear = nn.Linear(n_feature,1)

    def forward(self,x):
        y = self.linear(x)
        return y
# method 2
net = nn.Sequential(
    nn.Linear(num_inputs,1)
)
init.normal_(net[0].weight,mean=0,std=0.01)
init.constant_(net[0].bias,val=0)
loss = nn.MSELoss()
optimizer = optim.SGD(net.parameters(),lr=0.03)
# 思考adam为什么这里表现没sgd好
# optimizer = optim.Adam(net.parameters(),lr=0.03)

num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))