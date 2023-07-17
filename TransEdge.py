import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl
import random
from dgl.nn import GraphConv
import os

file_path = "TransEdge" # your absolute path

n_devices = 307
n_servers = 6

# server_capacities = np.array([730, 50, 100, 50, 300, 1000])#1230 -->smaller

# server_capacities = np.array([765, 50, 100, 50, 300, 1000])#1265 -->big_smaller

server_capacities = np.array([800, 50, 100, 50, 300, 1000])#1300 -->normal
# het
# server_capacities = np.array([50, 100, 800, 300, 50, 1000])#1300 -->het_normal

# server_capacities = np.array([800, 50, 135, 50, 300, 1000])#1335 -->smalle_bigger

# server_capacities = np.array([800, 50, 170, 50, 300, 1000])#1370 -->bigger

class Environment:
    def __init__(self):
        self.n_devices = n_devices
        self.n_servers = n_servers
        self.server_capacities = server_capacities
        self.state = None
        self.task_allocation = None
        self.total_delay = None
        self.total_delaywithPenalty = None
        #self.reset()

    def reset(self, flow):
        self.state = flow # np.ones(self.n_devices)
        self.server_capacities = np.copy(server_capacities)
        self.task_allocation = np.zeros((self.n_devices, self.n_servers))
        self.total_delay = np.zeros(self.n_servers)
        self.total_delaywithPenalty = np.zeros(self.n_servers)
        self.reward = np.array([0])
        return np.concatenate((self.state, self.server_capacities/max(server_capacities), self.reward))

    def step(self, action, device_id, greedyFalg, minDistanceMatrix, BasedStation, OptimalPlacement):
        server_id = action 
        server_id_backup = action
        delayPenalty = 0 
        cloudPenalty = 50 
        ForwardScale = 0.05 
        delayPenaltyIncrease = 0.03 
        ForwardCount = 0
                
        if self.server_capacities[server_id] >= self.state[device_id]: 

            self.server_capacities[server_id] -= self.state[device_id]
            self.task_allocation[device_id, server_id] += self.state[device_id]

            if server_id == (n_servers - 1): 
                self.total_delay[server_id] += self.state[device_id]/server_capacities[server_id]*cloudPenalty
                self.total_delaywithPenalty[server_id] += self.state[device_id]/server_capacities[server_id]*cloudPenalty
                
            else:
                self.total_delaywithPenalty[server_id] += self.state[device_id]/server_capacities[server_id] + minDistanceMatrix[BasedStation[server_id]][BasedStation[OptimalPlacement[device_id]]] * self.state[device_id]/1000*ForwardScale
                self.total_delay[server_id] += self.state[device_id]/server_capacities[server_id]
                if minDistanceMatrix[BasedStation[server_id]][BasedStation[OptimalPlacement[device_id]]] != 0:
                    ForwardCount += 1
            self.state[device_id] = 0
        else:
            self.state[device_id] -= self.server_capacities[server_id]
            self.task_allocation[device_id, server_id] += self.server_capacities[server_id]

            if server_id == (n_servers - 1): 
                self.total_delay[server_id] += self.server_capacities[server_id]/server_capacities[server_id]*cloudPenalty
                self.total_delaywithPenalty[server_id] += self.server_capacities[server_id]/server_capacities[server_id]*cloudPenalty
            else:
                self.total_delay[server_id] += self.server_capacities[server_id]/server_capacities[server_id]
                self.total_delaywithPenalty[server_id] += self.server_capacities[server_id]/server_capacities[server_id] + minDistanceMatrix[BasedStation[server_id]][BasedStation[OptimalPlacement[device_id]]] * self.server_capacities[server_id]/1000*ForwardScale
                if minDistanceMatrix[BasedStation[server_id]][BasedStation[OptimalPlacement[device_id]]] != 0:
                    ForwardCount += 1
            self.server_capacities[server_id] = 0
            if greedyFalg == 'withgreedy': 
                ForwardFlag = True 
                while ForwardFlag:
                    delayPenalty += delayPenaltyIncrease 
                    if sum(self.server_capacities[:n_servers - 1]) != 0: 
                        nonzeroIndex = np.nonzero(self.server_capacities[:n_servers - 1])[0] #
                        maxIndex = np.argmax(server_capacities[nonzeroIndex]) 
                        max_server_id = nonzeroIndex[maxIndex] 

                        server_id = max_server_id 
                        if self.server_capacities[server_id] >= self.state[device_id]: 
                            self.server_capacities[server_id] -= self.state[device_id]
                            self.task_allocation[device_id, server_id] += self.state[device_id]

                            if server_id == (n_servers - 1): 
                                self.total_delay[server_id] += self.state[device_id]/server_capacities[server_id]*cloudPenalty
                                self.total_delaywithPenalty[server_id] += self.state[device_id]/server_capacities[server_id]*cloudPenalty #
                                
                            else: 
                                self.total_delaywithPenalty[server_id] += self.state[device_id]/server_capacities[server_id] + minDistanceMatrix[BasedStation[server_id]][BasedStation[server_id_backup]] * self.state[device_id]/1000*ForwardScale
                                self.total_delay[server_id] += self.state[device_id]/server_capacities[server_id]
                                if minDistanceMatrix[BasedStation[server_id]][BasedStation[OptimalPlacement[device_id]]] != 0:
                                    ForwardCount += 1
                            self.state[device_id] = 0 
                            ForwardFlag = False 
                        else: 
                            self.state[device_id] -= self.server_capacities[server_id]
                            self.task_allocation[device_id, server_id] += self.server_capacities[server_id]

                            if server_id == (n_servers - 1):
                                self.total_delay[server_id] += self.server_capacities[server_id]/server_capacities[server_id]*cloudPenalty
                                self.total_delaywithPenalty[server_id] += self.server_capacities[server_id]/server_capacities[server_id]*cloudPenalty
                            else:
                                self.total_delay[server_id] += self.server_capacities[server_id]/server_capacities[server_id]
                                self.total_delaywithPenalty[server_id] += self.server_capacities[server_id]/server_capacities[server_id] + minDistanceMatrix[BasedStation[server_id]][BasedStation[server_id_backup]] * self.server_capacities[server_id]/1000*ForwardScale
                                
                                if minDistanceMatrix[BasedStation[server_id]][BasedStation[server_id_backup]] != 0:
                                    ForwardCount += 1
                            server_id_backup = server_id
                            self.server_capacities[server_id] = 0
                    
                    else: 
                        nonzeroIndex = np.nonzero(self.state)[0]
                        for NI in nonzeroIndex:
                            self.server_capacities[n_servers - 1] -= self.state[NI]
                            self.task_allocation[NI, n_servers - 1] += self.state[NI]
                            self.total_delay[n_servers - 1] += self.state[NI]/server_capacities[n_servers - 1]*cloudPenalty
                            self.total_delaywithPenalty[n_servers - 1] += self.state[NI]/server_capacities[n_servers - 1]*cloudPenalty
                            self.state[NI] = 0
                        ForwardFlag = False 

        self.reward = -np.sum(self.total_delay)/np.sum(self.task_allocation) - delayPenalty
        done = np.sum(self.state) == 0

        return np.concatenate((self.state, self.server_capacities/max(server_capacities), np.array([-self.reward]))), self.reward, done, self.task_allocation, self.total_delay, self.total_delaywithPenalty

class ActorCritic(nn.Module):
    def __init__(self, n_devices, n_servers):
        super(ActorCritic, self).__init__() 
        self.conv1 = GraphConv(2624, 128) 

        self.conv2 = GraphConv(128, 6) 

        self.commonCov = nn.Linear(n_devices + n_devices*n_servers + n_servers + 1, 128)
        
        self.actor = nn.Linear(128, n_servers)
        self.critic = nn.Linear(128, 1)

    def forward(self, g, input, inputs, flag = None):

        input = (input - np.min(input))/(np.max(input)-np.min(input))
        x = F.relu(self.conv1(g, torch.tensor(input)))
        x = self.conv2(g, x)
        x = x.reshape(-1)
        inputs = torch.cat([inputs, x], dim=0)
        x = torch.relu(self.commonCov(inputs))

        actor = self.actor(x)
        critic = self.critic(x)
        return actor, critic

model = ActorCritic(n_devices, n_servers)
optimizer = optim.Adam(model.parameters())

env = Environment()
import pandas as pd

Req_pd = np.load(os.path.join(file_path,"OriginData", "PEMSD4.npz"))
flow = Req_pd['truth'][:, 0, :, 0]
flow = np.transpose(flow)
scale_flow = 10 
flow = scale_flow * (flow - np.min(flow))/(np.max(flow) - np.min(flow))
flow_mean = np.mean(flow)

avg_flow = flow.copy()
row_flow = np.mean(avg_flow, axis=1)

row_flow = (row_flow - np.min(row_flow))/(np.max(row_flow) - np.min(row_flow))

avg_loss_list = []
avg_reward_list = []
sum_time_list = []
device_server_ID = []

minDistanceMatrix = np.load(os.path.join(file_path, "Dataset", "pems04_min_distance_matrix.npy"))
scale = 50 
minDistanceMatrix = scale * (minDistanceMatrix - np.min(minDistanceMatrix))/(np.max(minDistanceMatrix) - np.min(minDistanceMatrix))

OptimalPlacement =  np.loadtxt(os.path.join(file_path, "Dataset", "AdaptiveBestDNA_pesm04.txt"), dtype=np.int)

BasedStation = np.loadtxt(os.path.join(file_path,"Dataset", "BasedStation_pesm04.txt"), dtype=np.int)

N = flow.shape[1]

cap_size = 'normal' # 'smaller'，'big_smaller', 'normal', 'smalle_bigger', 'bigger' 
if cap_size in ['smaller','big_smaller', 'normal', 'smalle_bigger', 'bigger' ]:
    print("cap_size yes")
else: 
    np.show()
Flag = 'GCN'
greedyFalg = 'withgreedy'
g = None 

adj_matrix = np.load(os.path.join(file_path,"Dataset","pems04_distance_matrix.npy"))
adj_matrix = torch.tensor(adj_matrix)
g = dgl.DGLGraph()
g.add_nodes(adj_matrix.shape[0])

rows, cols = torch.nonzero(adj_matrix, as_tuple=True)
weights = adj_matrix[rows, cols]
weights = (weights - torch.min(weights))/(torch.max(weights)-torch.min(weights))
g.add_edges(rows, cols)
g.edata['weight'] = weights

import time
start_time = time.time()
N = 2 
for i in range(0, N): # Training iterations
    ReqNum = flow[:n_devices, i] 
    TransDelay = 0 
    for n_devices_j in range(0, n_devices): 
        TransDelay += minDistanceMatrix[n_devices_j][BasedStation[OptimalPlacement[n_devices_j]]]*ReqNum[n_devices_j]

    print("iteration =", i)
    state = torch.FloatTensor(env.reset(flow[:n_devices, i]))
    print("initial state =", state, sum(state[:n_devices]), type(state))
    done = False
    loss_list = []
    reward_list = []
    time_list_each = []
    for device_id in range(n_devices):  # Iterate over all devices
        device_server_i = []
        if (state[:n_devices] == 0).all().item():
            done = True
            break
        if state[device_id] != 0: 
            x = avg_flow
            optimizer.zero_grad()
            action_logits, value = model(g, x, state, Flag) 
            action_probs = nn.functional.softmax(action_logits, dim=0)
            action_probs = action_probs.detach().numpy()
            action = np.random.choice(np.arange(n_servers), p=action_probs)
            
            device_server_i.append(device_id)
            device_server_i.append(action)
            device_server_i.append(i)
            device_server_ID.append(device_server_i)
           
            next_state, reward, done, task_allocation, total_delay, total_delaywithPenalty = env.step(action, device_id, greedyFalg, minDistanceMatrix, BasedStation, OptimalPlacement)
            next_state = torch.FloatTensor(next_state)
           
            x = avg_flow
            _, next_value = model(g, x, next_state, Flag)
            

            td_target = reward + 0.99 * next_value
            delta = td_target - value

            action_log_probs = nn.functional.log_softmax(action_logits, dim=0)
            actor_loss = -action_log_probs[action] * delta.detach()
            critic_loss = delta.pow(2)

            loss = actor_loss + critic_loss
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().numpy())
            reward_list.append(reward)
            

    if len(loss_list) != 0: #
        avg_loss = np.mean(loss_list)
        avg_reward = np.mean(reward_list)
        avg_loss_list.append(avg_loss)
        avg_reward_list.append(avg_reward)
        
        time_list_each.append(TransDelay)
        time_list_each.append(sum(total_delay)*1000)
        time_list_each.append(TransDelay + sum(total_delay)*1000)
        time_list_each.append(sum(total_delaywithPenalty)*1000)

        sum_time_list.append(time_list_each)

        print("total iteration=", N, "solution", Flag, greedyFalg)
        print("transmission latency=", TransDelay)
        print("computation latency=", total_delay, sum(total_delay))
        print("computation latency with Penalty=", total_delaywithPenalty, sum(total_delaywithPenalty))
        print("total response latency=", TransDelay + sum(total_delay)*1000)

end_time = time.time()

run_time = end_time - start_time
print("total time =", run_time, "s")
np.savetxt(os.path.join(file_path, "sum_time_list.txt"), sum_time_list, fmt='%d')
np.savetxt(os.path.join(file_path, "device_server_ID.txt"), device_server_ID, fmt='%d')
np.savetxt(os.path.join(file_path, "avg_reward_list.txt"), avg_reward_list)

y=avg_reward_list 
x=[i for i in range(len(avg_reward_list))]

plt.plot(x,y)
plt.show()
