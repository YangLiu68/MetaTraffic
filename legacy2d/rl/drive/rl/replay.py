import numpy as np

class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int = 200_000):
        self.size = size
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros((size,), dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros((size,), dtype=np.float32)

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        if o2 is not None:
            self.obs2[self.ptr] = o2
        self.done[self.ptr] = d
        self.ptr += 1
        if self.ptr >= self.size:
            self.ptr = 0
            self.full = True

    def sample(self, batch_size: int):
        max_i = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_i, size=batch_size)
        return (self.obs[idx], self.act[idx], self.rew[idx], self.obs2[idx], self.done[idx])
