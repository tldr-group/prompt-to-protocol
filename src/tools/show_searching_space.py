import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import os

def contains_rnn_keywords(network):
    for module in network.modules():
        if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
            return True
    return False

def preprocess_input(tensor, model):
    if contains_rnn_keywords(model):
        # Adjust the tensor shape to (batch_size, sequence_length, input_size)
        tensor = tensor.unsqueeze(0)  # Adding batch dimension (1, sequence_length, input_size)
    return tensor


def show_searching_space(nn_module, gif_path):
    # NeuralNetwork = nn_module.NeuralNetwork
    model = NeuralNetwork()

    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Total parameters:", total_params)

    # if total_params > 30:
    #     print("The model has more than 30 parameters. Please reduce the model size.")
    #     return

    def initialize_weights_uniform(model, lower_bound=-1, upper_bound=1):
        for param in model.parameters():
            if param.dim() > 1:  
                nn.init.uniform_(param, lower_bound, upper_bound)
            else: 
                nn.init.uniform_(param, lower_bound, upper_bound)

    def generate_gif():
        fig, ax = plt.subplots()
        t_values = np.linspace(0, 3600, 1000).reshape(-1, 1)  # Shape: (3600, 1)
        t_tensor = torch.tensor(t_values, dtype=torch.float32)
        
        def update(frame):
            initialize_weights_uniform(model, -1, 1)
            preprocessed_tensor = preprocess_input(t_tensor, model)
            output = model(preprocessed_tensor).detach().numpy()
            ax.clear()
            ax.plot(t_values.flatten(), output.flatten())
            ax.set_xlim(0, 3600)
            ax.set_ylim(0, 10)
            ax.set_title('Neural Network Output')
            ax.set_xlabel('t')
            ax.set_ylabel('Output')
            return ax

        ani = FuncAnimation(fig, update, frames=50, repeat=True)
        ani.save(gif_path, writer='imagemagick')
        plt.close()

        print(f"GIF saved at {gif_path}")

    generate_gif()

# Check the number of parameters to ensure it is less than 30

if __name__ == "__main__":


       
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.fc1 = nn.Linear(1, 3, bias=False)
            self.bn1 = nn.BatchNorm1d(3)
            self.fc2 = nn.Linear(3, 1, bias=False)
        
        def forward(self, input):
            if input.dim() == 3: input = input.view(-1, input.shape[-1])
            x = self.fc1(input)
            x = self.bn1(x) 
            x = torch.relu(x) 
            x = self.fc2(x)
            output = torch.sigmoid(x) * 10 
            return output

    
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "GIFs"
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"{now}.gif")

    show_searching_space(NeuralNetwork, gif_path)