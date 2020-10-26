import numpy as np
import plotly.graph_objects as go
import copy
from agent import Agent

# input params
N = 50 #agents
K = 6 #days
b = 4 #optimal patrons
weeks = 1000 #iterations
alpha = 0.5 #monte carlo update
eps = 0.01 #exploration frequency
reward_type = 'global' # global, local, difference
ci = 0 #difference counterfactual

# payoff for single day(action)
def local_reward(day_count):
  return day_count * np.exp(-day_count/b)

# payoff for entire week
def system_reward(week):
  rew = 0
  for day_count in week:
    rew += local_reward(day_count)
  return rew

# contribution to system reward
def difference_reward(day_num, week):
  mod_week = copy.deepcopy(week)
  mod_week[day_num] += ci - 1 #remove contribution of agent, add counterfactual
  return system_reward(week) - system_reward(mod_week)

# gets reward dependent on constant
def get_reward(week, day_num):
  if reward_type == 'global':
    return system_reward(week)
  elif reward_type == 'local':
    return local_reward(week[day_num])
  elif reward_type == 'difference':
    return difference_reward(day_num, week)

# computes general alignment (bc all agents have same reward fn)
def alignment(week):
  res = 0
  for i in range(K):
    mod_week = copy.deepcopy(week)
    mod_week[i] += 1
    res += np.sign((get_reward(week, i)-get_reward(mod_week, i))*(system_reward(week)-system_reward(mod_week)))
  res /= K
  return res

rew_fig = go.Figure()
# run for ea. reward type
for reward_type in ['local', 'global', 'difference']:

  # create agents
  agents = []
  for i in range(N):
    agents.append(Agent(K, alpha, eps))

  # game loop
  rec_rew = []
  for week in range(weeks):
    # create bar
    bar = [0] * K

    # all agents take action, update system state
    for agent in agents:
      bar[agent.act()] += 1

    # assign reward
    for agent in agents:
      agent.reward(get_reward(bar, agent.day))
    
    # save system parameters
    rec_rew.append(system_reward(bar))
  rew_fig.add_trace(go.Scatter(y=rec_rew, name=reward_type.capitalize()))

  hist_fig = go.Figure()
  hist_fig.add_trace(go.Bar(y=bar))
  hist_fig.update_layout(
    font_family = 'Computer Modern',
    xaxis_title = 'Night',
    yaxis_title = 'Attendees',
    width=500,
    height=500
  )
  hist_fig.show()

# plot reward
rew_fig.update_layout(
  font_family="Computer Modern",
  xaxis_title="Iterations",
  yaxis_title="System Reward",
  width=500,
  height=500
)
rew_fig.show()

