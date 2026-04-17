# import gradio as gr
# import random
# import time

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox()
#     clear = gr.ClearButton([msg, chatbot])

#     def respond(message, chat_history):
#         bot_message = random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
#         chat_history.append({"role": "user", "content": message})
#         chat_history.append({"role": "assistant", "content": gr.Textbox(label="第一部分内容", value="第一部分默认文字", lines=2)})
#         time.sleep(2)
#         return "", chat_history

#     msg.submit(respond, [msg, chatbot], [msg, chatbot])

# demo.launch()



import gradio as gr

history = [
    {"role": "assistant", "content": "I am happy to provide you that report and plot."},
    {"role": "assistant", "content": gr.Textbox(label="第一部分内容", value="第一部分默认文字", lines=2)}
]

with gr.Blocks() as demo:
    gr.Chatbot(history)

demo.launch()