import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, ALPHA, n_action):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 200) # [number of input nodes, number of output nodes]
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,n_action) # coarse grain the action space

        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # send network to device

    def forward(self, observation):
        observation = T.Tensor(observation).to(self.device)
        # TODO reshape here
        observation = F.relu(self.fc1(observation))
        observation = F.relu(self.fc2(observation))
        actions = self.fc3(observation)
        # TODO reshape here

        return actions


class Agent(object):
    def __init__(self, gamma, epsilon, alpha,
                 maxMemoerySize, epsEnd=0.05,
                 replace=10000, *actionSpace):

        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_END = epsEnd
        self.actionSpace = actionSpace
        self.memSize = maxMemorySize
        self.steps = 0
        self.learn_step_counter = 0 # keep track of how many times agent calls the learn function (target network replacement)
        self.memory = [] # cheaper to turn list to numpy array than to stack numpy array
        self.memCntr = 0 # memory counter: total memory stored
        self.replace_target_cnt = replace # how often we replace the target network
        self.Q_eval = DeepQNetwork(alpha) # agent est of current set of states
        self.Q_next = DeepQNetwork(alpha) # agent est of successor set of states

    def storeTransition(self, state, action ,reward, state_):
        if self.memCentr < self.memSize:
            self.memory.append([state,action, reward, state_])
        else:
            self.memory[self.memCentr%self.memSize] = [state, action, reward, state_]
        self.memCentr  += 1

    def chooseAction(self, observation):
        rand = np.random.random() # for epsilon greedy
        actions = self.Q_eval.forward(observation)
        if rand < 1 - self.EPSILON:
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.actionSpace)
        self.steps += 1
        return action

    def learn(self, batch_size):
        # good subsampling to avoid correlation
        self.Q_eval.optimizer.zero_grad() # for batch optimization instead of full opt
        if self.replace_target_cnt is not None and\
           self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

        if self.memCntr + batch_size < self.memSize:
            memStart = int(np.random.choice(range(self.memCntr)))
        else:
            memStart = int(np.random.choice(range(self.memCntr-batch_size-1)))
        miniBatch = self.memory[memStart:memStart+batch_size]
        memory = np.array(miniBatch)

        Qpred = self.Q_eval.forward(list(memory[:, 0][:])).to(self.Q_eval.device)
        Qnext = self.Q_next.forward(list(memory[:, 3][:])).to(self.Q_eval.device)

        maxA = T.argmax(Qnext, dim=1).to(self.Q_eval.device)
        rewards = T.Tensor(list(memory[:, 2])).to(self.Q_eval.device)
        Qtarget = Qpred
        Qtarget[:,maxA] = rewards + self.GAMMA*T.max(Qnext[1])

        if self.steps > 500:
            if self.EPSILON - 1e-4 > self.EPS_END:
                self.EPSILON -= 1e-4
            else:
                self.EPSILON = self.EPS_END

        loss = self.Q_eval.loss(Qtarget, Qpred).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.learn_step_counter += 1



