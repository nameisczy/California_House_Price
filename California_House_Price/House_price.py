import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入数据
train_data = pd.read_csv("kaggle/California_House_Price/train.csv")
test_data = pd.read_csv("kaggle/California_House_Price/test.csv")

# 数据预处理
# 去掉label、用处不大的数据
all_features = pd.concat((train_data.iloc[:, 4:-1], test_data.iloc[:, 3:-1]))

# 处理日期
all_features['Last Sold On'] = pd.to_datetime(all_features['Last Sold On'], format='%Y-%m-%d')
all_features['Listed On'] = pd.to_datetime(all_features['Listed On'], format='%Y-%m-%d')

all_features['Last Sold Year'] = all_features['Last Sold On'].dt.year
all_features['Last Sold Month'] = all_features['Last Sold On'].dt.month
all_features['Last Sold Day'] = all_features['Last Sold On'].dt.day
all_features['Listed Year'] = all_features['Listed On'].dt.year
all_features['Listed Month'] = all_features['Listed On'].dt.month
all_features['Listed Day'] = all_features['Listed On'].dt.day

all_features = all_features.drop(['Last Sold On', 'Listed On'], axis=1)

# 标准化数据
numeric_features = all_features.dtypes[all_features.dtypes == 'float64'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 处理文本数据
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.astype(float)

# 数据分回训练集和测试集
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)
train_labels = torch.tensor(train_data['Sold Price'].values.reshape(-1, 1), dtype=torch.float32).to(device)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(train_features.shape[1], 256)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.out(x)
        return x

# 损失函数
loss = nn.MSELoss()

def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    return torch.sqrt(2 * loss(torch.log(clipped_preds), torch.log(labels)).mean())

def train(net, train_features, train_labels, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls = []
    train_dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(train_dataset, batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            l = log_rmse(net, X, y)
            l.backward()
            optimizer.step()
        net.eval()
        train_ls.append(log_rmse(net, train_features, train_labels).item())
        torch.cuda.empty_cache()  # 清理缓存
    return train_ls

def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = Net().to(device)
    train_ls = train(net, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
    plt.plot(np.arange(1, num_epochs + 1), train_ls, label='train')
    plt.xlabel('epoch')
    plt.ylabel('log rmse')
    plt.yscale('log')
    plt.legend()
    plt.show()
    print(f'最终训练log rmse: {float(train_ls[-1]):f}')

    net.eval()
    preds = net(test_features).detach().cpu().numpy()

    test_data['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['Sold Price']], axis=1)
    submission.to_csv('submission.csv', index=False)

num_epochs, lr, weight_decay, batch_size = 100, 0.1, 0.1, 128
train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)
