import torch.nn as nn
import torch.nn.functional as F


class MultiLabelLayer(nn.Module):
    def __init__(self,  class_num, input_length):
        super(MultiLabelLayer, self).__init__()
        self.class_num = class_num
        self.input_length = input_length
        self.softmax = nn.Softmax()
        linear_sq = []
        for i in range(len(class_num)):
            linear_sq.append(nn.Linear(self.input_length, class_num[i]))
        self.linear_sq = linear_sq

    def forward(self, x):
        result = []
        for linear in self.linear_sq:
            out = F.softmax(linear(x), dim=1)
            result.append(out)
        return result


class MultiOutModel(nn.Module):
    def __init__(self, model, label_list):
        super(MultiOutModel, self).__init__()
        self.model = model
        self.label_list = label_list

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        length = x.size(1)
        linear = MultiLabelLayer(self.label_list, length)
        out = linear(x)
        return out


if __name__ == '__main__':
    import torch
    from models.shufflenet import ShuffleNetG2
    model = ShuffleNetG2()
    net = MultiOutModel(model, [2, 2, 2])
    x = torch.rand(1, 3, 32, 32)
    print(net(x))

