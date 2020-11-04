# simulation of a mixed probability game
import numpy as np
import plotly.graph_objects as go

# input params
iterations = 50
alpha = 0.3
eps = 0.1


# monte carlo value update
def update_value(value, reward):
  return value + alpha * (reward - value)

# returns index of input choices, with random motion
def act(values):
  if np.random.uniform() > eps:
    # return highest value
    return values.index(max(values))
  else:
    return np.random.randint(0, len(values))

# local definition of reward
def reward(p1_choice, p2_choice):
  poss = [[3, 0], [0, 1]]
  return poss[p1_choice][p2_choice]

# data to save
rewards = [0] * iterations

# game, p1 moving randomly and p2 "learning"
p2_values = [0, 0]
for rnd in range(iterations):
  # both players choose
  p1_choice = np.random.randint(0,2)
  p2_choice = act(p2_values)
  # let p2 do value update
  p2_values[p2_choice] = update_value(p2_values[p2_choice], reward(p1_choice, p2_choice))
  # save data
  rewards[rnd] = reward(p1_choice, p2_choice)

# evaluate policy
rew_fig = go.Figure()
rew_fig.add_trace(go.Scatter(y=rewards))
rew_fig.update_layout(
  font_family="Computer Modern",
  xaxis_title="Iterations",
  yaxis_title="Player 2 Reward",
  width=500,
  height=500
)
rew_fig.show()

hist_fig = go.Figure()
hist_fig.add_trace(go.Bar(y=p2_values))
hist_fig.update_layout(
  font_family = 'Computer Modern',
  xaxis_title = 'Choice',
  yaxis_title = 'Learned Value',
  width=500,
  height=500
)
hist_fig.show()