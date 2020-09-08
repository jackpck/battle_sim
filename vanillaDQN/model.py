import torch
import torch.nn as nn
import torch.autograd as autograd


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16,32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, state):
        qvals = self.fc(state)
        return qvals

if __name__ == '__main__':
    import numpy as np

    batch_size = 5
    state_dim = 4
    naction = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DQN(state_dim, naction).to(device)

    states = np.random.uniform(0,1,size=(batch_size,state_dim))
    actions = np.random.randint(0,naction, size=batch_size)
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    Q = model.forward(states).gather(1, actions.unsqueeze(1))
    dones = np.array([False, False, True, True, True])
    dones = np.array([int(done) for done in dones])
    dones = torch.FloatTensor(dones).to(device)

    print('states.shape: ',states.shape)
    print('actions.shape: ',actions.shape)
    print('model.forward(states).shape: ',model.forward(states).shape)
    print('Q.shape: ',Q.shape)

    print('model.forward(states): ', model.forward(states))
    print('actions: ',actions.unsqueeze(1))
    print('Q: ',Q)

