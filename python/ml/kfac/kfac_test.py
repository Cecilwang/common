import unittest
import warnings

import torch
import torch.nn as nn

from python.ml.kfac import KFAC
from python.ml.kfac import classification_sampling


class TestKFAC(unittest.TestCase):

    def test_forward_and_backward(self):
        model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.ReLU(),
                              nn.Linear(2, 2, bias=False))
        for x in model:
            if not hasattr(x, "weight"):
                continue
            data = [float(i) for i in range(1, torch.numel(x.weight.data) + 1)]
            data = torch.tensor(data).reshape(x.weight.data.shape)
            x.weight = nn.Parameter(data)

        loss = nn.CrossEntropyLoss()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            kfac = KFAC(model.parameters())
            kfac.register(model)

            x = torch.tensor([[0.7, 0.5, 0.9], [0.1, 0.2, 0.3]])
            y = torch.tensor([0, 1])
            kfac.hook_on = True
            output = model(x)
            sampled_y = classification_sampling(output)
            loss(output, sampled_y).backward()

        # Sometimes it will fail because the sampling is different.
        model[0].weight.gtG = torch.tensor([[2.0450e-08, 2.0456e-08],
                                            [2.0456e-08, 2.0463e-08]])
        model[0].weight.gtA = torch.tensor([[0.2500, 0.1850, 0.3300],
                                            [0.1850, 0.1450, 0.2550],
                                            [0.3300, 0.2550, 0.4500]])
        model[2].weight.gtG = torch.tensor([[5.1035e-09, -5.1065e-09],
                                            [-5.1065e-09, 5.1095e-09]])
        model[2].weight.gtA = torch.tensor([[10.6600, 25.7800],
                                            [25.7800, 62.3650]])
        for p in model.parameters():
            torch.testing.assert_close(p.gtG, p.G)
            torch.testing.assert_close(p.gtA, p.A)


if __name__ == '__main__':
    unittest.main()
