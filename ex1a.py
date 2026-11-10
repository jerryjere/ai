from openai import OpenAI

client = OpenAI()

print("URK23CS1218")

description = input("Describe the function: ")

try:
    prompt = f"""
Generate only valid Python code.

The code must contain a Python function that satisfies this description:

{description}

Do not include explanations.
Do not include markdown.
Return only the Python function.
"""

    response = client.responses.create(
        model="gpt-5",
        input=prompt
    )

    code = response.output_text.strip()

    print("\n===== GPT Generated Code =====\n")
    print(code)

except Exception as e:
    print("\nOpenAI API is unavailable.")
    print("Reason:", e)

    print("\nSwitching to Offline Function Generator...\n")

    d = description.lower()

    if "add" in d:
        code = """
def add(a, b):
    return a + b
"""

    elif "subtract" in d:
        code = """
def subtract(a, b):
    return a - b
"""

    elif "multiply" in d:
        code = """
def multiply(a, b):
    return a * b
"""

    elif "divide" in d:
        code = """
def divide(a, b):
    return a / b
"""

    elif "square" in d:
        code = """
def square(x):
    return x * x
"""

    elif "cube" in d:
        code = """
def cube(x):
    return x * x * x
"""

    elif "factorial" in d:
        code = """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)
"""

    else:
        code = """
def hello():
    return "Function not available"
"""

    print(code)


exec(code)

print("\n===== Testing Function =====")

d = description.lower()

if "add" in d:
    result = add(10, 20)
    expected = 30

elif "subtract" in d:
    result = subtract(20, 5)
    expected = 15

elif "multiply" in d:
    result = multiply(4, 5)
    expected = 20

elif "divide" in d:
    result = divide(20, 4)
    expected = 5

elif "square" in d:
    result = square(8)
    expected = 64

elif "cube" in d:
    result = cube(3)
    expected = 27

elif "factorial" in d:
    result = factorial(5)
    expected = 120

else:
    result = hello()
    expected = "Function not available"

print("Output   :", result)
print("Expected :", expected)

if result == expected:
    print("\nTest Passed")
else:
    print("\nTest Failed")