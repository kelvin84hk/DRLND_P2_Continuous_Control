import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed,fc1_units=64,fc2_units=64,fc3_units=64): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """    
        super(Actor, self).__init__()  
        self.seed = torch.manual_seed(seed)    
        self.fc1=nn.Linear(state_size,fc1_units)
        self.b1=nn.BatchNorm1d(fc1_units)
        self.fc2=nn.Linear(fc1_units,fc2_units)
        self.b2=nn.BatchNorm1d(fc2_units) 
        self.fc3=nn.Linear(fc2_units,fc3_units)
        self.b3=nn.BatchNorm1d(fc3_units)
        self.fc4=nn.Linear(fc3_units,action_size)
        #self.tanh1=nn.Tanh()
        #self.tanh2=nn.Tanh()
        #self.tanh3=nn.Tanh()
        self.tanh4=nn.Tanh()

    def forward(self, state):
        
        x=F.relu(self.fc1(state))
        x=self.b1(x)
        x=F.relu(self.fc2(x))
        x=self.b2(x)
        x=F.relu(self.fc3(x))
        x=self.b3(x)
        x=self.tanh4(self.fc4(x))
                
        return x        

class CriticD4PG(nn.Module):
    """Critic (distribution) Model."""

    def __init__(self, state_size, action_size, seed,fc1_units=128,fc2_units=128,fc3_units=128,fc4_units=128,n_atoms=51, v_min=-1, v_max=1): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimesion of each action
            seed (int): Random seed
        """    
        super(CriticD4PG, self).__init__()  
        self.seed = torch.manual_seed(seed)    
        self.fc1=nn.Linear(state_size,fc1_units)
        self.b1=nn.BatchNorm1d(fc1_units)
        self.fc2=nn.Linear(fc1_units,fc2_units)
        self.b2=nn.BatchNorm1d(fc2_units)
        self.fc3=nn.Linear(fc2_units+action_size,fc3_units)
        self.b3=nn.BatchNorm1d(fc3_units)
        self.fc4=nn.Linear(fc3_units,fc4_units)
        self.fc5=nn.Linear(fc4_units,n_atoms)
        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max+delta , delta))

    def forward(self, state,action):
        
        xs=F.relu(self.fc1(state))
        xs=self.b1(xs)
        xs=F.relu(self.fc2(xs))
        xs=self.b2(xs)
        x = torch.cat((xs, action), dim=1)
        x=F.relu(self.fc3(x))
        x=self.b3(x)
        #x=F.relu(self.fc4(x))
        x=self.fc5(x)

        return x

    def distr_to_q(self, distr):
    	
    	weights=F.softmax(distr, dim=1)*self.supports
    	res=weights.sum(dim=1)
    	return res.unsqueeze(dim=-1)

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed,fc1_units=64,fc2_units=64,fc3_units=64,fc4_units=64): 
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimesion of each action
            seed (int): Random seed
        """    
        super(Critic, self).__init__()  
        self.seed = torch.manual_seed(seed)    
        self.fc1=nn.Linear(state_size,fc1_units)
        self.b1=nn.BatchNorm1d(fc1_units)
        self.fc2=nn.Linear(fc1_units,fc2_units)
        self.b2=nn.BatchNorm1d(fc2_units)
        self.fc3=nn.Linear(fc2_units+action_size,fc3_units)
        self.b3=nn.BatchNorm1d(fc3_units)
        self.fc4=nn.Linear(fc3_units,fc4_units)
        self.fc5=nn.Linear(fc4_units,1)
    
    def forward(self, state,action):
        
        xs=F.relu(self.fc1(state))
        #xs=self.b1(xs)
        xs=F.relu(self.fc2(xs))
        #xs=self.b2(xs)
        x = torch.cat((xs, action), dim=1)
        x=F.relu(self.fc3(x))
        #x=self.b3(x)
        x=F.relu(self.fc4(x))
        x=self.fc5(x)

        return x                


