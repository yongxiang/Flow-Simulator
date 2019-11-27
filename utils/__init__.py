from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LOOP = 1000
