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

    Repeat this pattern for 10 different networks.

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

    # ========== Test Mode: Check if OpenAI API is configured ==========
    print("=" * 60)
    print("LLM Network Generation (Case 3) - Test Mode")
    print("=" * 60)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print(f"\n✓ OpenAI API key is configured (length: {len(api_key)} chars)")
        print("\nYou can run the actual generation functions:")
        print("  - generate_initialization(): Generate 10 initial network structures")
        print("  - generate_new_network(seed1, seed2): Generate new network from two seeds")
    else:
        print("\n✗ OpenAI API key is NOT configured")
        print("  Set it with: export OPENAI_API_KEY='your-api-key'")
    
    # ========== Test Example Network Structures (Case 3: 3 inputs, output in [-8, -3]) ==========
    print("\n" + "-" * 60)
    print("Testing example neural network structures for Case 3...")
    print("Case 3: 3 inputs (V, T, SoC) -> 1 output (Current in [-8, -3])")
    print("-" * 60)
    
    import torch
    import torch.nn as nn
    import pybamm
    import numpy as np
    
    # Example network following the Case 3 template
    class SeedNetwork1(nn.Module):
        """MLP with LayerNorm and residual connection"""
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
            h_res = self.fc2(h) + h  # Residual connection
            h = self.act(h_res)
            out = -torch.sigmoid(self.fc3(h)) * 5 - 3  # Output in [-8, -3]
            return out
    
    # Example network 2: Simple MLP
    class SeedNetwork2(nn.Module):
        """Simple MLP with Tanh activation"""
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 4)
            self.ln1 = nn.LayerNorm(4, elementwise_affine=False)
            self.fc2 = nn.Linear(4, 1)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = torch.tanh(self.ln1(self.fc1(x)))
            out = -torch.sigmoid(self.fc2(h)) * 5 - 3  # Output in [-8, -3]
            return out
    
    # PyBaMM symbolic function example
    def nn_in_pybamm_example(X, Ws, bs):
        """Example PyBaMM symbolic implementation for a simple network"""
        W1, W2 = [w.tolist() for w in Ws]
        b1, b2 = [b.tolist() for b in bs]
        hidden = len(b1)
        
        # Layer 1: affine
        lin1 = []
        for i in range(hidden):
            x = pybamm.Scalar(b1[i])
            for j in range(3):
                x += pybamm.Scalar(W1[i][j]) * X[j]
            lin1.append(x)
        
        # LayerNorm (no affine)
        mean1 = sum(lin1) / hidden
        var1 = sum((x - mean1) ** 2 for x in lin1) / hidden
        eps = 1e-5
        inv_std1 = 1 / pybamm.sqrt(var1 + pybamm.Scalar(eps))
        ln1 = [(x - mean1) * inv_std1 for x in lin1]
        
        # Activation (tanh)
        h1 = [pybamm.tanh(x) for x in ln1]
        
        # Output layer
        lin2 = pybamm.Scalar(b2[0])
        for j in range(hidden):
            lin2 += pybamm.Scalar(W2[0][j]) * h1[j]
        
        # Sigmoid + scaling to [-8, -3]
        y = 1 / (1 + pybamm.exp(-lin2))
        return -pybamm.Scalar(5) * y - 3
    
    # Test the networks
    net1 = SeedNetwork1()
    net2 = SeedNetwork2()
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nSeed Network 1 (MLP + LayerNorm + Residual):")
    print(f"  Structure: 3 -> 3 -> 3 -> 1")
    print(f"  Parameters: {count_parameters(net1)}")
    print(f"  Constraint check: {'✓ PASS' if count_parameters(net1) < 35 else '✗ FAIL'} (< 35)")
    
    print(f"\nSeed Network 2 (Simple MLP + Tanh):")
    print(f"  Structure: 3 -> 4 -> 1")
    print(f"  Parameters: {count_parameters(net2)}")
    print(f"  Constraint check: {'✓ PASS' if count_parameters(net2) < 35 else '✗ FAIL'} (< 35)")
    
    # Test forward pass with 3 inputs (simulating V, T, SoC normalized)
    test_input = torch.tensor([[0.5, 0.3, 0.7]])  # 3 inputs
    
    try:
        output1 = net1(test_input)
        output2 = net2(test_input)
        
        print(f"\nForward pass test:")
        print(f"  Input (V_norm, T_norm, SoC_norm): [{test_input[0, 0]:.2f}, {test_input[0, 1]:.2f}, {test_input[0, 2]:.2f}]")
        print(f"  SeedNetwork1 output (Current): {output1.item():.4f} A")
        print(f"  SeedNetwork2 output (Current): {output2.item():.4f} A")
        
        # Verify output range [-8, -3]
        assert -8 <= output1.item() <= -3, f"Output1 {output1.item():.4f} out of range [-8, -3]!"
        assert -8 <= output2.item() <= -3, f"Output2 {output2.item():.4f} out of range [-8, -3]!"
        
        print(f"\n  Output range check: ✓ PASS (both in [-8, -3])")
        
        # Test PyBaMM symbolic function
        print(f"\nPyBaMM symbolic function test:")
        Ws = [net2.fc1.weight.detach().numpy(), net2.fc2.weight.detach().numpy()]
        bs = [net2.fc1.bias.detach().numpy(), net2.fc2.bias.detach().numpy()]
        X_pybamm = [pybamm.Scalar(0.5), pybamm.Scalar(0.3), pybamm.Scalar(0.7)]
        
        pybamm_output = nn_in_pybamm_example(X_pybamm, Ws, bs)
        print(f"  PyBaMM symbolic output type: {type(pybamm_output).__name__}")
        print(f"  ✓ PyBaMM symbolic function created successfully")
        
        print("\n" + "=" * 60)
        print("TEST PASSED - Network structures are valid!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED - Error occurred!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    # ========== Optional: Run actual LLM generation ==========
    print("\n" + "-" * 60)
    print("Optional: Run LLM generation")
    print("-" * 60)
    
    run_llm_test = False  # Set to True to test actual LLM generation
    
    if run_llm_test and api_key:
        print("\nRunning generate_new_network() with seed networks...")
        try:
            new_code = generate_new_network(seed1_code, seed2_code)
            print(f"\nGenerated response length: {len(new_code)} chars")
            print("LLM generation test completed!")
        except Exception as e:
            print(f"LLM generation failed: {e}")
    else:
        if not api_key:
            print("\nSkipping LLM test (no API key)")
        else:
            print("\nSkipping LLM test (set run_llm_test=True to enable)")
    
    print("\n" + "=" * 60)
    print("Test finished.")
    print("=" * 60)
