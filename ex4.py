import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
print("URK23CS1218")
model_name = "gpt2"
print("Loading GPT-2 model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()
print("GPT-2 loaded successfully!")
def get_completion_only(prompt, max_new_tokens=50):
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )
    return completion
prompt = """def calculate_average(numbers):
"""
print("\nGenerated code:")
print(get_completion_only(prompt))
def code_completion_interface(code):
    if not code.strip():
        return "Please enter some Python code."
    return get_completion_only(
        code,
        max_new_tokens=80
    )
demo = gr.Interface(
    fn=code_completion_interface,
    inputs=gr.Textbox(
        lines=15,
        label="Enter your Python code"
    ),
    outputs=gr.Textbox(
        lines=15,
        label="AI Code Completion"
    ),
    title="Transformer-Based Code Completion",
    description="Enter partial Python code and generate the next lines using GPT-2."
)
demo.launch()