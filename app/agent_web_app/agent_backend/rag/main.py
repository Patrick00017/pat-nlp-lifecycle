import gradio as gr
from bts_rag import response
import time

def real_response(message, history):
        try:
            print(message)
            nothink = response(message)
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            response_result = "Interrupted. Goodbye!"
            # break
        except Exception as e:
            nothink = f"Error: {e}"
            # print("Please try again or type 'quit' to exit")
        finally:
            for i in range(len(nothink)):
                time.sleep(0.0001)
                yield "" + nothink[: i+1]

gr.ChatInterface(
    real_response,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(placeholder="Ask me anything.", container=False, scale=7),
    title="Log Analysis",
    description="Ask question about system log",
    examples=[
        "query glue set function events in the system and return events list, start time is 2026-01-08 14:03:50.690, and end time is 2026-01-08 15:03:50.690",
        "track the material P.-.-.8.J lifecycle",
    ],
    save_history=False
    # cache_examples=True,
).launch(theme="ocean")