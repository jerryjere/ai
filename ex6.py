
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

print("URK23CS1218")

code = '''"""Example AI-generated code."""


def add_numbers(a: int, b: int) -> int:
    """Return the sum of two numbers."""
    return a + b
'''

print("\nAI-GENERATED CODE QUALITY EVALUATION")
print("\n1. SOURCE CODE\n" + code)

print("\n2. SYNTAX CHECK")
try:
    ast.parse(code)
    print("Syntax is valid.")
except SyntaxError as e:
    print("Syntax Error:", e)
    sys.exit(1)

with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "generated_code.py"
    path.write_text(code, encoding="utf-8")

    print("\n3. PYLINT STATIC ANALYSIS")
    pylint = subprocess.run(
        [sys.executable, "-m", "pylint", str(path)],
        capture_output=True, text=True, check=False
    )
    print(pylint.stdout + pylint.stderr)

    print("\n4. MYPY TYPE CHECKING")
    mypy = subprocess.run(
        [sys.executable, "-m", "mypy", str(path)],
        capture_output=True, text=True, check=False
    )
    print("Passed:", mypy.returncode == 0)
    print(mypy.stdout + mypy.stderr)
