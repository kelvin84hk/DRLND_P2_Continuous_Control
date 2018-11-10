import numpy as np
import random
import copy
from collections import namedtuple, deque

from p2_model import Actor,CriticD4PG,Critic
from prioritized_memory import Memory
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Vmax = 5
Vmin = 0
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,BUFFER_SIZE = int(1e5),BATCH_SIZE = 64,GAMMA = 0.99,TAU = 1e-3,LR_ACTOR = 1e-4,LR_CRITIC = 1e-4,WEIGHT_DECAY = 0.0001,UPDATE_EVERY = 4,IsPR=False,N_step=1,IsD4PG_Cat=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.BUFFER_SIZE=BUFFER_SIZE
        self.BATCH_SIZE=BATCH_SIZE
        self.GAMMA=GAMMA
        self.TAU=TAU

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.UPDATE_EVERY=UPDATE_EVERY
        self.N_step=N_step
        self.IsD4PG_Cat=IsD4PG_Cat
        self.rewards_queue=deque(maxlen=N_step)
        self.states_queue=deque(maxlen=N_step)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, seed).to(device)
        self.actor_target = Actor(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        if IsD4PG_Cat:
            self.critic_local = CriticD4PG(state_size, action_size, seed,n_atoms=N_ATOMS,v_min=Vmin,v_max=Vmax).to(device)
            self.critic_target = CriticD4PG(state_size, action_size, seed,n_atoms=N_ATOMS,v_min=Vmin,v_max=Vmax).to(device)
        else:    
            self.critic_local = Critic(state_size, action_size, seed).to(device)
            self.critic_target = Critic(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        
        # Replay memory
        self.BATCH_SIZE=BATCH_SIZE
        self.IsPR=IsPR
        if IsPR:
            self.memory = Memory(BUFFER_SIZE) # prioritized experienc replay
        else:    
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)

        # Noise process
        self.noise = OUNoise(action_size, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.train_start = 2000
        
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory        
        if self.IsPR:
            self.states_queue.appendleft([state,action])
            self.rewards_queue.appendleft(reward*self.GAMMA**self.N_step)
            for i in range(len(self.rewards_queue)):
                self.rewards_queue[i] = self.rewards_queue[i]/self.GAMMA

            if len(self.rewards_queue)>=self.N_step: # N-steps return: r= r1+gamma*r2+..+gamma^(t-1)*rt  
                temps=self.states_queue.pop()
                state = torch.tensor(temps[0]).float().to(device)
                next_state = torch.tensor(next_state).float().to(device)
                action = torch.tensor(temps[1]).float().unsqueeze(0).to(device)
                
                if self.IsD4PG_Cat:
                    self.critic_local.eval()
                    with torch.no_grad():
                        Q_expected = self.critic_local(state, action)
                    self.critic_local.train()
                    self.actor_target.eval()
                    with torch.no_grad():
                        action_next = self.actor_target(next_state)
                    self.actor_target.train()
                    self.critic_target.eval()
                    with torch.no_grad():
                        Q_target_next = self.critic_target(next_state, action_next)
                        Q_target_next =F.softmax(Q_target_next, dim=1)
                    self.critic_target.train()
                    sum_reward=torch.tensor(sum(self.rewards_queue)).float().unsqueeze(0).to(device)
                    done_temp=torch.tensor(done).float().to(device)
                    Q_target_next=self.distr_projection(Q_target_next,sum_reward,done_temp,self.GAMMA**self.N_step)
                    Q_target_next = -F.log_softmax(Q_expected, dim=1) * Q_target_next
                    error  = Q_target_next.sum(dim=1).mean().cpu().data
                else:
                    self.critic_local.eval()
                    with torch.no_grad():
                        Q_expected = self.critic_local(state, action).cpu().data
                    self.critic_local.train()   
                    action_next = self.actor_target(next_state)
                    Q_target_next = self.critic_target(next_state, action_next).squeeze(0).cpu().data    
                    Q_target = sum(self.rewards_queue) + ((self.GAMMA**self.N_step)* Q_target_next * (1 - done))
                    error = abs(Q_target-Q_expected)
                state=state.cpu().data.numpy()
                next_state=next_state.cpu().data.numpy()
                action=action.squeeze(0).cpu().data.numpy()
                self.memory.add(error, (state, action, sum(self.rewards_queue), next_state, done))
                self.rewards_queue.pop()
                if done:
                    self.states_queue.clear()
                    self.rewards_queue.clear()

            self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
            if self.t_step == 0:
                # If enough samples are available in memory, get random subset and learn
                if self.memory.tree.n_entries > self.train_start:
                    batch_not_ok=True
                    while batch_not_ok:
                        mini_batch, idxs, is_weights = self.memory.sample(self.BATCH_SIZE)
                        mini_batch = np.array(mini_batch).transpose()
                        if mini_batch.shape==(5,self.BATCH_SIZE):
                            batch_not_ok=False
                        else:    
                            print(mini_batch.shape)    
                    try:
                        states = np.vstack([m for m in mini_batch[0] if m is not None])
                    except:
                        #print([m.shape for m in mini_batch[0] if m is not None])
                        print('states not same dim')
                        pass
                    try:    
                        actions = np.vstack([m for m in mini_batch[1] if m is not None])
                    except:
                        print('actions not same dim')
                        pass
                    try:    
                        rewards = np.vstack([m for m in mini_batch[2] if m is not None])
                    except:
                        print('rewars not same dim')
                        pass
                    try:
                        next_states = np.vstack([m for m in mini_batch[3] if m is not None])
                    except:
                        print('next states not same dim')
                        pass
                    try:
                        dones = np.vstack([m for m in mini_batch[4] if m is not None])
                    except:
                        print(mini_batch.shape)
                        print(mini_batch[4].shape)
                        print([m for m in mini_batch[4] if m is not None])
                        print('dones not same dim')
                        pass
                    # bool to binary
                    dones = dones.astype(int)

                    states = torch.from_numpy(states).float().to(device)
                    actions = torch.from_numpy(actions).float().to(device)
                    rewards = torch.from_numpy(rewards).float().to(device)
                    next_states = torch.from_numpy(next_states).float().to(device)
                    dones = torch.from_numpy(dones).float().to(device)
                    experiences=(states, actions, rewards, next_states, dones)
                    self.learn(experiences, self.GAMMA, idxs)

        else :
           
            self.states_queue.appendleft([state,action])
            self.rewards_queue.appendleft(reward*self.GAMMA**self.N_step)
            for i in range(len(self.rewards_queue)):
                self.rewards_queue[i] = self.rewards_queue[i]/self.GAMMA

            if len(self.rewards_queue)>=self.N_step:  # N-steps return: r= r1+gamma*r2+..+gamma^(t-1)*rt
                temps=self.states_queue.pop()   
                self.memory.add(temps[0], temps[1], sum(self.rewards_queue), next_state, done)
                self.rewards_queue.pop()
                if done:
                    self.states_queue.clear()
                    self.rewards_queue.clear()
            # If enough samples are available in memory, get random subset and learn
            self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
            if self.t_step == 0:
                if len(self.memory) >self.BATCH_SIZE:
                    experiences = self.memory.sample()
                    self.learn(experiences, self.GAMMA)

    def act(self, state, add_noise=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        #state = torch.tensor(np.moveaxis(state,3,1)).float().to(device)
        state = torch.tensor(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.squeeze(np.clip(action,0.0,1.0))

    def distr_projection(self,next_distr_v, rewards_v, dones_mask_t, gamma):
        next_distr = next_distr_v.data.cpu().numpy()
        rewards = rewards_v.data.cpu().numpy()
        dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
        batch_size = len(rewards)
        proj_distr = np.zeros((batch_size, N_ATOMS), dtype=np.float32)
        dones_mask=np.squeeze(dones_mask)
        rewards = rewards.reshape(-1)

        for atom in range(N_ATOMS):
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * DELTA_Z) * gamma))
            b_j = (tz_j - Vmin) / DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            
            proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
            ne_mask = u != l
            
            proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
            proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

        if dones_mask.any():
            proj_distr[dones_mask] = 0.0
            tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones_mask]))
            b_j = (tz_j - Vmin) / DELTA_Z
            l = np.floor(b_j).astype(np.int64)
            u = np.ceil(b_j).astype(np.int64)
            eq_mask = u == l
            if dones_mask.shape==():
                if dones_mask:
                    proj_distr[0, l] = 1.0
                else:
                    ne_mask = u != l
                    proj_distr[0, l] = (u - b_j)[ne_mask]
                    proj_distr[0, u] = (b_j - l)[ne_mask]    
            else:
                eq_dones = dones_mask.copy()
                
                eq_dones[dones_mask] = eq_mask
                if eq_dones.any():
                    proj_distr[eq_dones, l[eq_mask]] = 1.0
                ne_mask = u != l
                ne_dones = dones_mask.copy()
                ne_dones[dones_mask] = ne_mask
                if ne_dones.any():
                    proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
                    proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

        return torch.FloatTensor(proj_distr).to(device)
    
    def learn(self, experiences, gamma,idxs =None):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
         # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        if self.IsD4PG_Cat:
            Q_targets_next =F.softmax(Q_targets_next, dim=1)
            Q_targets_next=self.distr_projection(Q_targets_next,rewards,dones,gamma**self.N_step)
            Q_targets_next = -F.log_softmax(Q_expected, dim=1) * Q_targets_next
            critic_loss = Q_targets_next.sum(dim=1).mean()
        else:        
            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        if self.IsPR:
            if self.IsD4PG_Cat:
                self.critic_local.eval()
                with torch.no_grad():
                    errors = Q_targets_next.sum(dim=1).cpu().data.numpy()
                self.critic_local.train()      
            else:    
                errors = torch.abs(Q_expected - Q_targets).squeeze(0).cpu().data.numpy()
        
            # update priority
            for i in range(self.BATCH_SIZE):
                idx = idxs[i]
                self.memory.update(idx, errors[i])

        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        if self.IsD4PG_Cat:
            crt_distr_v=self.critic_local(states, actions_pred)
            actor_loss = -self.critic_local.distr_to_q(crt_distr_v)
            actor_loss = actor_loss.mean()
        else:
            actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.TAU)                  

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
