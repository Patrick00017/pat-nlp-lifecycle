import gradio as gr
import requests

# API endpoints
CHAT_API = "http://127.0.0.1:8000/chat"
RESUME_API = "http://127.0.0.1:8000/resume"

def chat_with_agent(thread_id, message):
    """Send a message to the chat API."""
    payload = {"thread_id": thread_id, "message": message}
    response = requests.post(CHAT_API, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("thread_id"), data.get("response"), data.get("interrupt")
    else:
        return None, f"Error: {response.text}", None

def resume_conversation(thread_id, approved, modified_args):
    """Resume a conversation using the resume API."""
    payload = {
        "thread_id": thread_id,
        "approved": approved,
        "modified_args": modified_args,
    }
    response = requests.post(RESUME_API, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("response"), data.get("interrupt")
    else:
        return f"Error: {response.text}", None

def gradio_interface(thread_id, message, approved, modified_args):
    if message:
        thread_id, response, interrupt = chat_with_agent(thread_id, message)
        if interrupt:
            return thread_id, response, interrupt, "Interrupt occurred. Please handle it."
        return thread_id, response, None, ""
    elif thread_id and approved is not None:
        response, interrupt = resume_conversation(thread_id, approved, modified_args)
        if interrupt:
            return thread_id, response, interrupt, "Interrupt occurred. Please handle it."
        return thread_id, response, None, ""
    else:
        return thread_id, "", None, "Please provide a message or handle an interrupt."

with gr.Blocks() as demo:
    gr.Markdown("# Chat with Agent")

    with gr.Row():
        thread_id_input = gr.Textbox(label="Thread ID (optional)")
        message_input = gr.Textbox(label="Message")

    with gr.Row():
        approved_input = gr.Checkbox(label="Approved (for resume)")
        modified_args_input = gr.Textbox(label="Modified Args (JSON, for resume)")

    with gr.Row():
        submit_button = gr.Button("Submit")

    with gr.Row():
        thread_id_output = gr.Textbox(label="Thread ID", interactive=False)
        response_output = gr.Textbox(label="Response", interactive=False)
        interrupt_output = gr.Textbox(label="Interrupt", interactive=False)
        status_output = gr.Textbox(label="Status", interactive=False)

    submit_button.click(
        gradio_interface,
        inputs=[thread_id_input, message_input, approved_input, modified_args_input],
        outputs=[thread_id_output, response_output, interrupt_output, status_output],
    )

demo.launch()