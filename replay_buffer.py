import random
class replay_buffer(object):
    def __init__(self, max_size):
        self.buffer = [] 
        # why i use a list?, can i use arry?
        self.max_size = max_size

    def push(self, state, action, reward, next_state, Done):
        experience = (state, action, reward, next_state, Done)
        self.buffer.append(experience)
    
    def sample (self, batch_size):
        state_sample = []
        next_State_sample = []
        action_sample = []
        reward_sample = []
        done_sample = []
        
        sample = random.sample(self.buffer, batch_size)
        
        for experience in sample:
            state, action, reward, next_state, done = experience
            state_sample.append(state)
            action_sample.append(action)
            reward_sample.append(reward)
            next_State_sample.append(next_state)
            done_sample.append(done)
        return (state_sample, action_sample, reward_sample, next_State_sample, done_sample)
    
    def truncate(self):
        self.buffer = self.buffer[-self.max_size:] 

    def __len__(self):
        return len(self.buffer)

    
    