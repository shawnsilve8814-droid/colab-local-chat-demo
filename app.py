import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "gpt2"

def setup_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = setup_model()

if torch.cuda.is_available():
    model = model.to("cuda")


def generate_response(history, user_input, max_new_tokens=100):
    # history: list of (user, bot)
    prompt = ""
    for u, b in history:
        prompt += f"User: {u}\nAssistant: {b}\n"
    prompt += f"User: {user_input}\nAssistant:"

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.to("cuda")

    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_k=50, top_p=0.95)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip the prompt from the returned text
    response = text[len(prompt):].strip()
    return response


def respond(history, user_input):
    response = generate_response(history, user_input)
    history.append((user_input, response))
    # GRadio chat expects list of tuples
    chat_history = [(u, b) for u, b in history]
    return chat_history, history

with gr.Blocks() as demo:
    gr.Markdown("# Local GPT-2 Chat (Demo)")
    chat = gr.Chatbot()
    state = gr.State([])  # history
    txt = gr.Textbox(show_label=False, placeholder="Type your message and press Enter")

    txt.submit(respond, [state, txt], [chat, state])

    demo.launch()
