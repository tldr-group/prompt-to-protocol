import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# "gpt-4o-2024-05-13" "gpt-3.5-turbo-0125",
def generate_initialization():
# 
    def get_completion(messages, model="gpt-5-2025-08-07", 
                      ):

                completion = client.chat.completions.create(
                    model= model,
                    messages=messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens
                )
                return completion.choices[0].message.content

    system_message ="""
You are tasked with generating 10 PyTorch neural networks and its symbolic counterpart in PyBaMM including RNN, MLP, LSTM, etc. Follow the strict design constraints and code template provided below:

1. Framework:
You must import:
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   import pybamm
   import numpy as np
2. Output Scaling:
Ensure the final output is scaled to the range [-8, -3]. Use a sigmoid activation followed by scaling to achieve this.
3. Parameters Constraint:
Each neural network should have three input and a single output.
4. Layer Configuration:
Consider different architectures, MLP/RNN/LSTM/etc.
Consider different activation functions.
Please make sure using LayerNorm in MLP.
Do not use dropout layers.
The parameters of generated networks should be less than 35.

5. Network Naming:
Each generated pytorch class should be named NeuralNetwork. The PyBaMM symbolic version should be a standalone function named nn_in_pybamm
Below is an exmple and template for each network:

    # Neural Network 1
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch
    import numpy as np

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 3)
            self.ln1 = nn.LayerNorm(3, elementwise_affine=False)
            self.act = nn.LeakyReLU(0.01)
            self.fc2 = nn.Linear(3, 3)
            self.fc3 = nn.Linear(3, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.fc1(x)
            h = self.ln1(h)
            h = self.act(h)
            h_res = self.fc2(h) + h
            h = self.act(h_res)
            out = -torch.sigmoid(self.fc3(h)) * 5 - 3
            return out

    def nn_in_pybamm(X, Ws, bs):
        W1, W2, W3 = [w.tolist() for w in Ws]
        b1, b2, b3 = [b.tolist() for b in bs]
        hidden = len(b1)
        assert hidden == 3
        # --- layer 1: affine ---
        lin1 = []
        for i in range(hidden):
            x = pybamm.Scalar(b1[i])
            for j in range(3):
                x += pybamm.Scalar(W1[i][j]) * X[j]
            lin1.append(x)

        # --- LayerNorm (no affine) ---
        mean1 = sum(lin1) / hidden
        var1 = sum((x - mean1) ** 2 for x in lin1) / hidden
        eps = 1e-5  # small constant to avoid division by zero
        inv_std1 = 1 / pybamm.sqrt(var1 + pybamm.Scalar(eps))
        ln1 = [(x - mean1) * inv_std1 for x in lin1]

        # --- activation ---
        h1 = [pybamm.maximum(x, 0.01 * x) for x in ln1]

        # --- layer 2: affine + residual + activation ---
        lin2 = []
        for i in range(hidden):
            x = pybamm.Scalar(b2[i])
            for j in range(hidden):
                x += pybamm.Scalar(W2[i][j]) * h1[j]
            lin2.append(x)
        h2 = [
            pybamm.maximum(lin2[i] + h1[i], 0.01 * (lin2[i] + h1[i]))
            for i in range(hidden)
        ]

        # --- output layer + sigmoid + scaling ---
        lin3 = pybamm.Scalar(b3[0])
        for j in range(hidden):
            lin3 += pybamm.Scalar(W3[0][j]) * h2[j]
        y = 1 / (1 + pybamm.exp(-lin3))
        I_scale = 5  # Scale output to current range
        return -pybamm.Scalar(I_scale) * y -3
    ```


    # Neural Network 2
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pytorch
    import numpy as np

    class  NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            ...
            
        def forward(self, input):
            ...
            output = - torch.sigmoid(x) * 5 -3  # Scale output to range [-8, -3]
            return output

    def nn_in_pybamm(X, Ws, bs):
        ...
    return -pybamm.Scalar(I_scale) * y -3
    ```

    # Neural Network 3
    ```python
    ```

    # Neural Network 4
    ```python
    ```

    # Neural Network 5
    ```python
    ```

    # Neural Network 6
    ```python
    ```

    # Neural Network 7
    ```python
    ```

    # Neural Network 8
    ```python
    ```

    # Neural Network 9
    ```python
    ```

    # Neural Network 10
    ```python
    ```

    Make sure every Python code block is enclosed by matching triple backticks: use python to open and to close each block
    """


    messages = [
        {'role': 'system', 'content': system_message},
    ]
    print('=====================')

    print(messages)

    response = get_completion(messages)
    print('=====================')

    print(response)
    return response



def generate_new_network(seed1_code, seed2_code):

    def get_completion(messages, model="gpt-5-2025-08-07", 
                            # temperature=0.8, 
                            # max_tokens=1500
                            ):

                completion = client.chat.completions.create(
                    model= model,
                    messages=messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens
                )
                return completion.choices[0].message.content

    system_message ="""
    You are tasked with generating a PyTorch neural network and its symbolic counterpart in PyBaMM, based on two given example networks. Follow the strict design constraints and code template provided below:

    1. Framework:
    You must import:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import pybamm
        import numpy as np

    2. Output Scaling:
    Ensure the final output is scaled to the range [-8, -3]. Use a sigmoid activation followed by scaling to achieve this.

    3. Parameters Constraint:
    Each neural network should have three inputs and a single output.

    4. Layer Configuration:
    - Consider different architectures, MLP/RNN/LSTM/etc.
    - Number of hidden layers should range between 0 and 3.
    - Number of neurons per hidden layer should be between 1 and 4.
    - Use LayerNorm (without learnable affine parameters).
    - Do not use dropout layers.
    - Vary activation functions (e.g., ReLU, LeakyReLU, ELU, Tanh, etc.).
    - You may include residual (skip) connections.
    - The parameters of generated networks should be less than 35.

    5. Naming:
    - The PyTorch model class must be named `NeuralNetwork`.
    - The PyBaMM symbolic function must be named `nn_in_pybamm`.

    Below is the template to follow for each generated network:

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pybamm
    import numpy as np

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # define layers, layer norms, activations here

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # forward through layers, activations, residuals here
            out = torch.sigmoid(self.final_layer(x)) * 5 - 3
            return out

    def nn_in_pybamm(X, Ws, bs):
        # Unpack weights and biases
        # Build symbolic computation
        return -pybamm.Scalar(5) * y - 3
    ```
    """
#         
    user_message = f"""
    Generate a different neural network structure based on the two given neural networks.
    IMPORTANT: The parameters of generated networks should be less than 35.
    """



    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message},
        {'role': 'user', 'content': f"Seed 1: {seed1_code}"},
        {'role': 'user', 'content': f"Seed 2: {seed2_code}"},
    ]

    print(messages)

    response = get_completion(messages)
    return response




if __name__ == "__main__":
    seed1_code = """
    ```python
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
        x = self.fc1(input)
        x = self.bn1(x)  # Batch normalization
        x = torch.relu(x)  # Using ReLU activation function
        x = self.fc2(x)
        output = torch.sigmoid(x) * 5  # Scale output to range [0, 5]
        return output
```
    """

    seed2_code = """
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 3, bias=False)
        self.fc2 = nn.Linear(3, 2, bias=False)
        self.fc3 = nn.Linear(2, 1, bias=False)
        self.prelu = nn.PReLU()  # Using PReLU activation function
        self.layer_norm1 = nn.LayerNorm(3)  # Layer normalization
        self.layer_norm2 = nn.LayerNorm(2)  # Layer normalization
        
    def forward(self, input):
        x = self.fc1(input)
        x = self.layer_norm1(x)  
        x = self.prelu(x)  
        x = self.fc2(x)
        x = self.layer_norm2(x)  
        x = self.prelu(x) 
        x = self.fc3(x)
        output = torch.sigmoid(x) * 5  
        return output

```
    """

    new_code = generate_new_network(seed1_code, seed2_code)
    print(new_code)