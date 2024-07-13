import torch 
import torch.nn as nn
class Q_network(nn.Module):
    def __init__(self, input_dims, hidden_dims , output_dims):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, hidden_dims)
        self.fc4 = nn.Linear(hidden_dims, hidden_dims)
        self.fc5 = nn.Linear (hidden_dims, output_dims)
        self.relu = nn.ReLU()

    def forward(self, X): # note this name cannot be changed !
        a1 = self.relu(self.fc1(X.float()))
        a2 = self.relu(self.fc2(a1))
        a3 = self.relu(self.fc3(a2))
        a4 = self.relu(self.fc4(a3))
        a5 = self.fc5(a4)
        return a5

def get_input(snake, apple):
    dirs_checker = snake.getproximity()
    """This function is used to get the tensor form of the input"""
    # Convert the positions and directions to tensors and ensure they have the same number of dimensions
    snake_pos_tensor = torch.from_numpy(snake.pos).double().flatten()
    snake_dir_tensor = torch.from_numpy(snake.dir).double().flatten()
    apple_pos_tensor = torch.from_numpy(apple.pos).double().flatten()
    dirs_checker_tensor = torch.tensor(dirs_checker).double().flatten()
    
    # Concatenate the tensors
    X = torch.cat([snake_pos_tensor, snake_dir_tensor, apple_pos_tensor, dirs_checker_tensor])
    return X

