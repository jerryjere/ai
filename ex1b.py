import ollama
def generate_function(description):
    response = ollama.chat(
        model="llama3",
        messages=[{
            "role": "user",
            "content": (
                f"{description}\n"
                "Return only Python function code without markdown or explanation. "
                "Use list(range(n + 1))."
            )
        }],
        options={"temperature": 0}
    )

    code = response.message.content.strip()

    if code.startswith("```"):
        code = "\n".join(code.splitlines()[1:-1]).strip()

    return code


description = "Write a function named n_nat_numbers(n) that returns a list of numbers from 0 to n."

try:
    print("Natural Language Description:")
    print(description)

    code = generate_function(description)
    print("\nGenerated Function:")
    print(code)

    namespace = {}
    exec(code, namespace)
    func = namespace["n_nat_numbers"]

    test_cases = [
        (0, [0]),
        (1, [0, 1]),
        (3, [0, 1, 2, 3]),
        (5, [0, 1, 2, 3, 4, 5]),
        (10, list(range(11)))
    ]

    print("\nExecuting Tests...")

    for n, expected in test_cases:
        output = func(n)
        status = "PASS" if output == expected else "FAIL"
        print(f"{status} | Input: ({n},) | Output: {output} | Expected: {expected}")

except Exception as e:
    print("Error:", e)
    print("Check Ollama, llama3 installation, and generated code.")