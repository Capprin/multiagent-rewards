import numpy as np

class Agent:

  def __init__(self, K, alpha, eps):
    self.values = [0] * K #no known values
    self.alpha = alpha #learning rate
    self.eps = eps #exploration freq
    self.day = 0 #last chosen day

  # returns integer night to attend
  def act(self):
    if np.random.uniform() > self.eps:
      # use values
      for i, val in enumerate(self.values):
        if val > self.values[self.day]:
          self.day = i
    else:
      # explore; choose randomly
      self.day = np.random.randint(0,len(self.values))
    return self.day

  # does monte carlo value update
  def reward(self, reward):
    self.values[self.day] = self.values[self.day] + self.alpha*(reward-self.values[self.day])
