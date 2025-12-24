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
You are required to generate 10 battery charging control algorithms. The algorithm should calculate the charging current as a function of the battery's real-time voltage (V), state of charge (SoC), and temperature (T).
Follow the instructions below when designing the function:

Battery Specifications:

The voltage range should be between 3.0 V and 4.25 V, with an additional penalty applied for operation above 4.20 V.
Temperature is starting at 308.15 kelvin
Maximum allowed charging current: 8 A 
Minimum allowed charging current: 3 A
Charging time should be close to 0.5 hours.

Charging Current Function Requirements:

The charging current must be negative. For example, -5 means charging at 5 amperes. In the simulation, negative current indicates charging.
The absolute value of the charging current must not exceed 8 amperes.
Do not use any Boolean operators or logic statements such as "if", "and", "or", "not", or similar.
Since we’ll be using this function inside the PyBaMM library, please replace any `math` functions with their PyBaMM equivalents—for example, change `math.sin` to `pybamm.sin`. But do not use `pybamm.pi`, if you need pi, use `3.14` instead.
Please try differnt algorithms! Don't use the same or similar algorithm multiple times.
Please make sure all current function called `adaptive_current`.
Make sure every Python code block is enclosed by matching triple backticks: use python to open and to close each block
Make sure clamps the magnitude to [3, 8] A with pybamm.maximum/pybamm.minimum

Here are some examples of the charging function:
    # Charing Function 1
    ```python
    def adaptive_current(vars_dict):
        V   = vars_dict["Voltage [V]"]
        SOC = vars_dict["SoC"]
        T   = vars_dict["Volume-averaged cell temperature [K]"]
        v_pen = 1 / (1 + 15 * pybamm.maximum(0, V - 4.20))
        a = 5.6
        b1 = 1.4*(3.85 - V)
        b2 = 1.0*(0.55 - SOC)
        b3 = 1.2*(3.85 - V)*(0.55 - SOC)
        b4 = 0.6*(308.15 - T)/30
        desired = a + b1 + b2 + b3 + b4
        mag = pybamm.minimum(8, pybamm.maximum(3, desired * v_pen))
        return -mag
    ```


    # Charing Function 2
    ```python
    def adaptive_current(vars_dict):
        V = vars_dict["Voltage [V]"]              
        T = vars_dict["Volume-averaged cell temperature [K]"]
        SOC = vars_dict["SoC"]                   
        

    ```

    # Charing Function 3
    ```python
    def adaptive_current(vars_dict):


    ```

    # Charing Function 4
    ```python
    def adaptive_current(vars_dict):

    
    ```

    # Charing Function 5
    ```python
    def adaptive_current(vars_dict):

    ```

    # Charing Function 6
    ```python
    def adaptive_current(vars_dict):

    ```

    # Charing Function 7
    ```python
    def adaptive_current(vars_dict):

    ```

    # Charing Function 8
    ```python
    def adaptive_current(vars_dict):

    ```

    # Charing Function 9
    ```python
    def adaptive_current(vars_dict):

    ```

    # Charing Function 10
    ```python
    def adaptive_current(vars_dict):
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
You are required to generate a new battery charging control algorithm based on the two given algorithms. 
The algorithm should calculate the charging current as a function of the battery's real-time voltage (V), state of charge (SoC), and temperature (T).

Battery Specifications:

The voltage range should be between 3.0 V and 4.25 V, with an additional penalty applied for operation above 4.20 V.
Temperature is starting at 308.15 kelvin
Maximum allowed charging current: 8 A 
Minimum allowed charging current: 3 A
Charging time should be close to 0.5 hours.

Charging Current Function Requirements:

The charging current must be negative. For example, -5 means charging at 5 amperes. In the simulation, negative current indicates charging.
The absolute value of the charging current must not exceed 8 amperes.
Do not use any Boolean operators or logic statements such as "if", "and", "or", "not", or similar.
Since we’ll be using this function inside the PyBaMM library, please replace any `math` functions with their PyBaMM equivalents—for example, change `math.sin` to `pybamm.sin`. But do not use `pybamm.pi`, if you need pi, use `3.14` instead.
Please try differnt algorithms! Don't use the same or similar algorithm multiple times.
Please make sure all current function called `adaptive_current`.

Follow the format below when generating the algorithm:
    # Charing Function
    ```python
    def adaptive_current(vars_dict):
        V = vars_dict["Voltage [V]"]              
        T = vars_dict["Volume-averaged cell temperature [K]"]
        SOC = vars_dict["SoC"]                   
        

    ``
    Make sure Python code block is enclosed by matching triple backticks: use python to open and to close each block.
    Make sure clamps the magnitude to [3, 8] A with pybamm.maximum/pybamm.minimum
    
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




