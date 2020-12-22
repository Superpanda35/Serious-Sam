
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):

    """
    Deep Q-Learning Network class using Pytorch that will train the agent
    """

    #constructor
    def __init__(self,learning_rate, input_dims,fc1_dims,fc2_dims,n_actions):
        super(DeepQNetwork,self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions   #number of actions

        #create the deep learning layers
        self.fc1 = nn.Linear(np.prod(self.input_dims),self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.n_actions)

        #Adam optimizer
        self.optimizer = optim.Adam(self.parameters(),lr = learning_rate)

        #mean squared error
        self.loss = nn.MSELoss()

        self.device = T.device('cpu')
        self.to(self.device)


    def forward(self,state):
        """

        :param state: the observation or state of the agent right now
        :return: the probabilities associated with each of the actions
        """
        #flatten the observation matrix

        #flattening the batch -> (batch_size, obs_size, obs_size)
        if len(state.size()) > 2:
            obs = T.flatten(state,start_dim=1,end_dim=3).float()
        else:
            #individual observation
            obs = T.flatten(state).float()


        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent:

    #agent constructor
    def __init__(self, alpha, gamma, epsilon,input_dims,batch_size,n_actions,
                 max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-4):

        #hyperparameters
        self.alpha = alpha  #learning rate
        self.gamma = gamma  #discount factor
        self.epsilon = epsilon  #exploration factor
        self.input_dims = input_dims  #dimensions of the states
        self.eps_min = eps_end  #minimum that epsilon can be
        self.eps_dec = eps_dec  #how much to decrement epsilon by
        self.mem_size = max_mem_size  #how much are we going to store in memory
        self.action_space = [i for i in range(n_actions)]  #a list of the different actions possible
        self.batch_size = batch_size
        self.mem_cntr = 0

        #deep q network for this agent
        self.Q_eval = DeepQNetwork(self.alpha,n_actions = n_actions,input_dims = input_dims,
                                   fc1_dims = 64, fc2_dims = 64)

        # could also use a replay buffer to store past observations
        # Init replay buffer
        # replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

        #store the memory of the past states
        self.state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32)


        #to calculate the temporal difference
        self.new_state_memory = np.zeros((self.mem_size,*input_dims),dtype = np.float32)

        #store the past actions
        self.action_memory = np.zeros(self.mem_size,dtype=np.int32)

        #store the rewards achieved
        self.reward_memory = np.zeros(self.mem_size,dtype=np.float32)

        #to keep track when the mission is over and the game is over
        #to facilitate the q-values
        self.terminal_memory = np.zeros(self.mem_size,dtype=np.bool)

        self.q_table = dict()
        self.pastActions = []


    def store_transition(self, state, action, reward, new_state, done):
        """

        :param state: current state that the agent is on
        :param action: the action the agent is going to take
        :param reward: how much reward it will receive if it took that action
        :param new_state: the next state it will be after taking that action
        :param done: if the mission
        :return:
        """

        #since memory we are using is static and finite we just replace values when memory is full
        index = self.mem_cntr % self.mem_size

        #update our memory
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        #change the index of the next open position in our memory
        self.mem_cntr += 1


    def choose_action(self,observation) -> int:
        """

        Select action according to e-greedy policy

        Args:
            action_values: list of possible actions
            q_network (QNetwork): Q-Network
            epsilon (float): probability of choosing a random action

        Returns:
            action (int): chosen action [0, action_size)

        :param observation: the current observation from the environment
        :return: an int value that maps to a action in the action space
        """


        #choose the max q-value action
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            action = self.Q_eval.forward(state)
            action = T.argmax(action).item()

        #choose a e-greedy action
        else:
            action = np.random.choice(self.action_space)


        return action


    def learn(self):
        """
        using Deep Q Network have the agent learn
        update the Q-function

        """

        #if memory is too small, then return not much to update on our DQN
        if self.mem_cntr <= self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace = False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        #prepare the batch
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        #evaluate the actions that we took
        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch]

        #use target network here if needed
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next,dim=1)[0]

        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward()

        self.Q_eval.optimizer.step()

        #decrement self.epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self. eps_min


        #for measuring purposes return the loss

        return loss if loss!=None else 0











