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
