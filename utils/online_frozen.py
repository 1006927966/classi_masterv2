from abc import ABCMeta

import torch

class ClassifierApi(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwargs, ):
        super().__init__()
        self.device = kwargs.get('device') or 'cpu'
        self.model_path = kwargs.get('model_path') or 'semantic_frozen_model.pt'

        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._construct_model(self.model_path)

    def _construct_model(self, model_path):
        model = torch.jit.load(model_path, map_location=self.device)
        model.eval()
        return model

    def forward(self, x):
        y = self.model(x)
        y = torch.nn.functional.softmax(y, dim=1)[0]
        y = y.cpu()
        return y

if __name__ == '__main__':

    my_model= ClassifierApi(model_path='/defaultShare/share/wujl/83/master_models/resnext101_17mix/model_18_frozen.pt')
    example = torch.rand(1, 3, 224, 224)   # only for cpu and cmd (pycharm will get error output results)

    traced_script_module = torch.jit.trace(func=my_model, example_inputs=example)
    result = traced_script_module(example)
    print(result)
    traced_script_module.save('/defaultShare/share/wujl/83/master_models/resnext101_17mix/online_frozen_model.pt')