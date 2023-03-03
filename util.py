import numpy as np

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    for k, v in fr.items():
        if type(v) is dict and k in to and type(to[k]) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

'''
Common noise function for DDPG and control tasks
'''
class ActionNoise:
    def __init__(self, mu, theta=.2, sigma=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
