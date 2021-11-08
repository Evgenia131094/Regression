from torch import nn


class RegModel(nn.Module):
    def __init__(self, in_features, out_features=1, hidden_features=600):
        super(RegModel, self).__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc2_bn = nn.BatchNorm1d(hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc3_bn = nn.BatchNorm1d(hidden_features)
        self.fc4 = nn.Linear(hidden_features, hidden_features)
        self.fc4_bn = nn.BatchNorm1d(hidden_features)
        self.fc5 = nn.Linear(hidden_features, hidden_features)
        self.fc5_bn = nn.BatchNorm1d(hidden_features)
        self.fc6 = nn.Linear(hidden_features, hidden_features)
        self.fc6_bn = nn.BatchNorm1d(hidden_features)
        self.fc7 = nn.Linear(hidden_features, hidden_features)
        self.fc7_bn = nn.BatchNorm1d(hidden_features)
        self.fc8 = nn.Linear(hidden_features, out_features)
        self.fc8_bn = nn.BatchNorm1d(out_features)

        self.act = nn.ReLU()

        self.d = nn.Dropout(p=0.2)

    def forward(self, x):
        print(x.shape)
        x = self.d(self.act(self.fc1_bn(self.fc1(x))))
        x = self.d(self.act(self.fc2_bn(self.fc2(x))))
        x = self.d(self.act(self.fc3_bn(self.fc3(x))))
        x = self.d(self.act(self.fc4_bn(self.fc4(x))))
        # x = self.d(self.act(self.fc5_bn(self.fc5(x))))
        # x = self.d(self.act(self.fc6_bn(self.fc6(x))))
        # x = self.d(self.act(self.fc7_bn(self.fc7(x))))
        # for _ in range(self.hidden_layers):
        # x = self.d(self.act(self.fc2(x)))
        return self.fc8(x)


if __name__ == '__main__':
    print(RegModel(10))