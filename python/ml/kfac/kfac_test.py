import unittest
import warnings

import backpack as bp
import torch
import torch.nn as nn

from python.ml.kfac import KFAC
from python.ml.kfac import config

config.compatible_with_backpack = True


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
            kfac.register((model, loss))

            bp.extend(model)
            bp.extend(loss)

            x = torch.tensor([[0.7, 0.5, 0.9], [0.1, 0.2, 0.3]])
            y = torch.tensor([0, 1])
            with bp.backpack(bp.extensions.KFAC()):
                loss(model(x), y).backward()

        # Sometimes it will fail because the sampling is different.
        for p in model.parameters():
            torch.testing.assert_close(p.kfac[0], p.G)
            torch.testing.assert_close(p.kfac[1], p.A)


if __name__ == '__main__':
    unittest.main()
