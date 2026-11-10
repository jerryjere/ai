import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline


# 1. Real Python code samples
real_code = [
    """def bubble_sort(a):
    for i in range(len(a)):
        for j in range(len(a) - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a""",

    """def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a""",

    """def linear_search(a, x):
    for i, value in enumerate(a):
        if value == x:
            return i
    return -1""",

    """def binary_search(a, x):
    left, right = 0, len(a) - 1
    while left <= right:
        mid = (left + right) // 2
        if a[mid] == x:
            return mid
        if a[mid] < x:
            left = mid + 1
        else:
            right = mid - 1
    return -1""",

    """def find_max(a):
    maximum = a[0]
    for value in a:
        if value > maximum:
            maximum = value
    return maximum""",

    """def find_min(a):
    minimum = a[0]
    for value in a:
        if value < minimum:
            minimum = value
    return minimum""",

    """def reverse_list(a):
    return a[::-1]""",

    """def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",

    """def is_even(n):
    return n % 2 == 0""",

    """def sum_list(a):
    total = 0
    for value in a:
        total += value
    return total""",
]

# 2. Generate code through Ollama
prompts = [
    "Write a Python bubble sort function.",
    "Write a Python insertion sort function.",
    "Write a Python linear search function.",
    "Write a Python binary search function.",
    "Write a Python function to find the maximum in a list.",
    "Write a Python function to find the minimum in a list.",
    "Write a Python function to reverse a list.",
    "Write a Python factorial function.",
    "Write a Python function that checks if a number is even.",
    "Write a Python function that sums a list.",
]

print("\nGENERATOR: CodeLlama using Ollama")
generated_code = []
for i, prompt in enumerate(prompts, 1):
    try:
        response = ollama.generate(
            model="codellama",
            prompt="Return only Python code, with no markdown or explanation. " + prompt,
            options={"temperature": 0.2, "num_predict": 200},
        )
    except Exception as error:
        raise SystemExit(
            "Ollama generation failed. Check that Ollama is running "
            "and that 'ollama pull codellama' has completed.\n" + str(error)
        )
    code = response["response"].strip()
    if not code:
        raise SystemExit("CodeLlama returned empty code. Please run the program again.")
    generated_code.append(code)
    print(f"Generated sample {i}/{len(prompts)}")

# 3. Prepare train/test data (1 = Real, 0 = Generated)
X = real_code + generated_code
y = [1] * len(real_code) + [0] * len(generated_code)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Discriminator: TF-IDF + Logistic Regression
classifier = make_pipeline(
    TfidfVectorizer(token_pattern=r"(?u)\b\w+\b"),
    LogisticRegression(max_iter=1000),
)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print("\nDISCRIMINATOR RESULTS")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
print(classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Generated", "Real"],
    zero_division=0,
))

# 5. Predict one held-out real and one held-out generated sample
print("FINAL PREDICTIONS")
labels = {0: "Generated", 1: "Real"}
for actual_label in [1, 0]:
    index = y_test.index(actual_label)
    sample = X_test[index]
    prediction = int(classifier.predict([sample])[0])
    confidence = max(classifier.predict_proba([sample])[0])
    print(f"\nActual: {labels[actual_label]}")
    print(f"Code:\n{sample}")
    print(f"Prediction: {labels[prediction]}")
    print(f"Confidence: {confidence:.2f}")