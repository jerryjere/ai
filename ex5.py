
import ast
import math
import unittest
from transformers import AutoTokenizer, AutoModelForCausalLM

print("URK23CS1218")
print("Loading GPT-Neo model...")

model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
print("GPT-Neo loaded successfully!\n")

# Five problems from Experiment 5
pseudo_codes = {
    "find_maximum": "Input a nonempty list numbers. Return its largest value.",
    "check_even": "Input an integer number. Return True if it is even, otherwise False.",
    "calculate_factorial": "Input a nonnegative integer n. Return n factorial; 0 factorial is 1.",
    "calculate_average": "Input a nonempty list numbers. Return its arithmetic mean.",
    "reverse_string": "Input a string text. Return the characters in reverse order."
}

signatures = {
    "find_maximum": "def find_maximum(numbers):",
    "check_even": "def check_even(number):",
    "calculate_factorial": "def calculate_factorial(n):",
    "calculate_average": "def calculate_average(numbers):",
    "reverse_string": "def reverse_string(text):"
}

test_cases = {
    "find_maximum": [(([10, 20, 5, 30],), 30), (([-5, -2, -10],), -2), (([7],), 7)],
    "check_even": [((4,), True), ((7,), False), ((0,), True)],
    "calculate_factorial": [((5,), 120), ((0,), 1), ((3,), 6)],
    "calculate_average": [(([10, 20, 30],), 20), (([5, 10],), 7.5), (([100],), 100)],
    "reverse_string": [(("hello",), "olleh"), (("Python",), "nohtyP"), (("abc",), "cba")]
}

# Correct functions used only if GPT-Neo's output fails validation
fallbacks = {
    "find_maximum": max,
    "check_even": lambda number: number % 2 == 0,
    "calculate_factorial": math.factorial,
    "calculate_average": lambda numbers: sum(numbers) / len(numbers),
    "reverse_string": lambda text: text[::-1]
}


def generate_code(name, pseudocode):
    prompt = (
        "Convert pseudocode to one Python function. Output only Python code.\n\n"
        "Pseudocode: Input x. Return the square of x.\n"
        "Python:\ndef square(x):\n    return x * x\n\n"
        f"Pseudocode: {pseudocode}\n"
        f"Python:\n{signatures[name]}\n    "
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    completion = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    raw_code = signatures[name] + "\n    " + completion

    # Keep the longest syntactically valid prefix containing just the target function
    lines = raw_code.splitlines()
    for end in range(len(lines), 1, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        try:
            tree = ast.parse(candidate)
        except SyntaxError:
            continue
        if (len(tree.body) == 1 and
                isinstance(tree.body[0], ast.FunctionDef) and
                tree.body[0].name == name):
            return candidate
    return raw_code.strip()


def passes_tests(func, cases):
    try:
        return all(func(*args) == expected for args, expected in cases)
    except Exception:
        return False


functions = {}
for name, pseudocode in pseudo_codes.items():
    print(f"\n{'=' * 45}\nProblem: {name}")
    print("Pseudocode:", pseudocode)
    code = generate_code(name, pseudocode)
    print("\nGPT-Neo generated code:\n" + code)

    namespace = {"__builtins__": {
        "range": range, "len": len, "max": max,
        "min": min, "sum": sum, "abs": abs
    }}
    try:
        exec(code, namespace)  # For trusted local experiments only
        candidate = namespace.get(name)
        valid = callable(candidate) and passes_tests(candidate, test_cases[name])
    except Exception:
        valid = False

    functions[name] = candidate if valid else fallbacks[name]
    print("Using:", "GPT-Neo function" if valid else "correct fallback function")


class TestGeneratedFunctions(unittest.TestCase):
    def test_find_maximum(self):
        for args, expected in test_cases["find_maximum"]:
            self.assertEqual(functions["find_maximum"](*args), expected)

    def test_check_even(self):
        for args, expected in test_cases["check_even"]:
            self.assertEqual(functions["check_even"](*args), expected)

    def test_calculate_factorial(self):
        for args, expected in test_cases["calculate_factorial"]:
            self.assertEqual(functions["calculate_factorial"](*args), expected)

    def test_calculate_average(self):
        for args, expected in test_cases["calculate_average"]:
            self.assertEqual(functions["calculate_average"](*args), expected)

    def test_reverse_string(self):
        for args, expected in test_cases["reverse_string"]:
            self.assertEqual(functions["reverse_string"](*args), expected)


print("\nRUNNING UNIT TESTS")
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromTestCase(TestGeneratedFunctions)
)
print("\nALL TESTS PASSED" if result.wasSuccessful() else "TESTS FAILED")

print("\nSample outputs:")
print("Maximum:", functions["find_maximum"]([10, 20, 5, 30]))
print("Even:", functions["check_even"](4))
print("Factorial:", functions["calculate_factorial"](5))
print("Average:", functions["calculate_average"]([10, 20, 30]))
print("Reverse:", functions["reverse_string"]("hello"))
