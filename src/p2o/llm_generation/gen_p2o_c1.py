import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)


def generate_new_network(seed1_code, seed2_code):

    def get_completion(messages, model="gpt-5-2025-08-07", 
                            # temperature=0.8, max_tokens=1500
                            ):

                completion = client.chat.completions.create(
                    model= model,
                    messages=messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens
                )
                return completion.choices[0].message.content

    system_message ="""
    Based on the two given neural networks, which represent the shape of a charging protocol, make a small and focused adjustment to generate a new neural network. Follow these specific instructions:

    1. **Framework**: Use PyTorch. Import torch, torch.nn, and torch.nn.functional.
    2. **Output Scaling**: Ensure the final output is scaled to the range of 0 to 10.
    3. **Parameters Constraint**: The new neural network should have a single input and a single output.
    4. **Subtle Adjustments**: Refine the existing networks, not a complete redesign. Make a minimal change without reducing the number of parameters, unless necessary.
    5. **Parameter number**: The parameters of generated networks should be less than 35.
    6. **Structure**: Provide your answer in the following format:

    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class  NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            ...
            
        def forward(self, input):
            ...
            output = torch.sigmoid(x) * 10  # Scale output to range [0, 10]
            return output

    ```
    """
#         
    user_message = f"""
    Make a small and focused adjustment to generate a new neural network. Make sure the new network works properly.
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


def generate_initialization():

    def get_completion(messages, model="gpt-5-2025-08-07", 
                            # temperature=0.7, max_tokens=1500
                            ):

                completion = client.chat.completions.create(
                    model= model,
                    messages=messages,
                    # temperature=temperature,
                    # max_tokens=max_tokens
                )
                return completion.choices[0].message.content

    system_message ="""
    Generate 10 reasonable trainable (has parameters to learn) neural network structures using PyTorch with the following constraints:

1. Framework:
Use PyTorch. Import torch, torch.nn, and torch.nn.functional.
2. Output Scaling:
Ensure the final output is scaled to the range [0, 10]. Use a sigmoid activation followed by scaling to achieve this.
3. Parameters Constraint:
Each neural network should have a single input and a single output.
4. Layer Configuration:
Number of layers should range between 0 and 3.
Neurons in each layer should be between 1 and 4.
Consider use different activation functions.
Do not use dropout layers.
5. Network Naming:
Each generated class should be named NeuralNetwork. When creating instances of these classes, the name NeuralNetwork should always be used.
Below is a template for each network:

    # Neural Network 1
    ```python
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class  NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            ...
            
        def forward(self, input):
            ...
            output = torch.sigmoid(x) * 10  # Scale output to range [0, 10]
            return output

    ```

    # Neural Network 2
    ```python
    ...
    ```

    Repeat this pattern for 10 different networks.

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


# Example usage:
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    
    print("=" * 60)
    print("LLM Network Generation - Test Mode")
    print("=" * 60)
    
    # ========== Test Mode: Check if OpenAI API is configured ==========
    
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
    
    # ========== Test Example Network Structures ==========
    print("\n" + "-" * 60)
    print("Testing example neural network structures...")
    print("-" * 60)
    
    # Example seed network 1
    class SeedNetwork1(nn.Module):
        def __init__(self):
            super(SeedNetwork1, self).__init__()
            self.fc1 = nn.Linear(1, 4)
            self.fc2 = nn.Linear(4, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.sigmoid(self.fc2(x)) * 10
            return x
    
    # Example seed network 2
    class SeedNetwork2(nn.Module):
        def __init__(self):
            super(SeedNetwork2, self).__init__()
            self.fc1 = nn.Linear(1, 3)
            self.fc2 = nn.Linear(3, 2)
            self.fc3 = nn.Linear(2, 1)
        
        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x)) * 10
            return x
    
    # Test seed networks
    net1 = SeedNetwork1()
    net2 = SeedNetwork2()
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nSeed Network 1:")
    print(f"  Structure: 1 -> 4 -> 1")
    print(f"  Parameters: {count_parameters(net1)}")
    
    print(f"\nSeed Network 2:")
    print(f"  Structure: 1 -> 3 -> 2 -> 1")
    print(f"  Parameters: {count_parameters(net2)}")
    
    # Test forward pass
    test_input = torch.tensor([[0.5]])
    
    try:
        output1 = net1(test_input)
        output2 = net2(test_input)
        
        print(f"\nForward pass test:")
        print(f"  Input: {test_input.item():.4f}")
        print(f"  SeedNetwork1 output: {output1.item():.4f}")
        print(f"  SeedNetwork2 output: {output2.item():.4f}")
        
        # Verify output range [0, 10]
        assert 0 <= output1.item() <= 10, "Output1 out of range!"
        assert 0 <= output2.item() <= 10, "Output2 out of range!"
        
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
        print("\nRunning generate_initialization()...")
        try:
            response = generate_initialization()
            print(f"\nGenerated response length: {len(response)} chars")
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

