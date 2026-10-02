from __future__ import annotations

import unittest
import torch
from torch.utils.data import TensorDataset

from common import capture_rng, restore_rng
from train_force import make_loader


def epoch_order(loader):
    return torch.cat([batch[0] for batch in loader]).tolist()


class RecoveryRngTests(unittest.TestCase):
    def test_sampler_order_survives_worker_restart(self):
        dataset=TensorDataset(torch.arange(64))
        continuous_generator=torch.Generator().manual_seed(17)
        continuous=make_loader(dataset,8,True,continuous_generator,2)
        first=epoch_order(continuous); second=epoch_order(continuous)
        interrupted_generator=torch.Generator().manual_seed(17)
        interrupted=make_loader(dataset,8,True,interrupted_generator,2)
        self.assertEqual(epoch_order(interrupted),first)
        state=capture_rng(interrupted_generator)
        del interrupted
        resumed_generator=torch.Generator().manual_seed(999)
        resumed=make_loader(dataset,8,True,resumed_generator,2)
        restore_rng(state,resumed_generator)
        self.assertEqual(epoch_order(resumed),second)


if __name__=="__main__":unittest.main()
