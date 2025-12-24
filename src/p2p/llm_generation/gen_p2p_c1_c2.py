import openai
import os
import pybamm

openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# "gpt-4o-2024-05-13" "gpt-3.5-turbo-0125",
def generate_initialization():

    def get_completion(messages, model="gpt-5-2025-08-07", 
                           ):

                completion = client.chat.completions.create(
                    model= model,
                    messages=messages,
      
                )
                return completion.choices[0].message.content

    system_message ="""
You are required to generate 10 battery charging control algorithms to maintain constant heat generation rate.
The functions will be used with PyBaMM simulation library and must return PyBaMM expressions.

Battery Specifications:
Maximum allowed charging current: 10 A 
Minimum allowed charging current: 0 A

Charging Current Function Requirements:

The charging current must be negative. For example, -5 means charging at 5 amperes. In the simulation, negative current indicates charging.
The absolute value of the charging current must not exceed 10 amperes.
Do not use any Boolean operators or logic statements such as "if", "and", "or", "not", or similar.
Use PyBaMM functions and variables: use `pybamm.t` for time variable, `pybamm.sin`, `pybamm.cos`, `pybamm.exp`, `pybamm.tanh`, `pybamm.sqrt`, etc.
Do not use `pybamm.pi`, if you need pi, use `3.14` instead.
Please try different algorithms! Don't use the same or similar algorithm multiple times.
The function should be named `current_function` and should NOT take any parameters.
The function should return a PyBaMM expression that represents the charging current as a function of time.
Make sure every Python code block is enclosed by matching triple backticks: use python to open and to close each block

Here are some examples of the charging function:
    # Charging Function 1
    ```python
    def current_function():
        Q = 1.5  
        R0 = 0.02 
        t = pybamm.t
        R_t = R0 * (1 + 0.2 * pybamm.sin(2 * 3.14 * t / 3600))
        i_mag = pybamm.sqrt(Q / R_t)
        i = -10 * pybamm.tanh(i_mag / 10)
        return i
    ```

    # Charging Function 2
    ```python
    def current_function():


    ```

    # Charging Function 3
    ```python
    def current_function():


    ```

    # Charging Function 4
    ```python
    def current_function():


    ```

    # Charging Function 5
    ```python
    def current_function():
    ```

    # Charging Function 6
    ```python
    def current_function():

    ```

    # Charging Function 7
    ```python
    def current_function():

    ```

    # Charging Function 8
    ```python
    def current_function():

    ```

    # Charging Function 9
    ```python
    def current_function():

    ```

    # Charging Function 10
    ```python
    def current_function():

    ```

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



def generate_new_algorithm(seed1_code, seed2_code):

    def get_completion(messages, model="gpt-5-2025-08-07", 
                            ):

                completion = client.chat.completions.create(
                    model= model,
                    messages=messages,
                )
                return completion.choices[0].message.content

    system_message ="""
You are required to generate a new battery charging control algorithm based on the two given algorithms to maintain constant heat generation rate.
The function will be used with PyBaMM simulation library and must return PyBaMM expressions.

Battery Specifications:
Maximum allowed charging current: 10 A 
Minimum allowed charging current: 0 A

Charging Current Function Requirements:

The charging current must be negative. For example, -5 means charging at 5 amperes. In the simulation, negative current indicates charging.
The absolute value of the charging current must not exceed 10 amperes.
Do not use any Boolean operators or logic statements such as "if", "and", "or", "not", or similar.
Use PyBaMM functions and variables: use `pybamm.t` for time variable, `pybamm.sin`, `pybamm.cos`, `pybamm.exp`, `pybamm.tanh`, `pybamm.sqrt`, etc.
Do not use `pybamm.pi`, if you need pi, use `3.14` instead.
Please try different algorithms! Don't use the same or similar algorithm multiple times.
The function should be named `current_function` and should NOT take any parameters.
The function should return a PyBaMM expression that represents the charging current as a function of time.

Follow the format below when generating the algorithm:
    # Charging Function
    ```python
    def current_function():
        t = pybamm.t
        # Your algorithm here
        return i  # where i is the charging current expression
    ```
    Make sure Python code block is enclosed by matching triple backticks: use python to open and to close each block.
    
    """
#         
    user_message = f"""
    Generate a different charging algorithms based on the two given algorithm.
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




