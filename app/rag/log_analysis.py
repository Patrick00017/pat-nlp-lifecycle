import llama_cpp
import outlines
import httpx
from enum import Enum
from pydantic import BaseModel, Field
from typing import Union
from outlines import Template
import json
import datetime

llm = llama_cpp.Llama(
    "D:/code/gguf-models/hermes8b/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf",
    # tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
    #     "NousResearch/Hermes-2-Pro-Llama-3-8B"
    # ),
    n_gpu_layers=-1,
    flash_attn=True,
    n_ctx=8192,
    verbose=False,
)
model = outlines.from_llamacpp(llm)


class Reason_and_Act(BaseModel):
    Scratchpad: str = Field(
        ...,
        description="Information from the Observation useful to answer the question",
    )
    Thought: str = Field(
        ...,
        description="It describes your thoughts about the question you have been asked",
    )


class Final_Answer(BaseModel):
    Scratchpad: str = Field(
        ...,
        description="Information from the Observation useful to answer the question",
    )
    Final_Answer: str = Field(
        ..., description="Answer to the question grounded on the Observation"
    )


class Decision(BaseModel):
    Decision: Union[Reason_and_Act, Final_Answer]


json_schema = Decision.model_json_schema()
print(f"json schema: {json_schema}")

hermes_prompt = Template.from_file("./prompt.txt")


class ChatBot:
    def __init__(self, prompt=""):
        self.prompt = prompt

    def __call__(self, user_prompt):
        self.prompt += user_prompt
        result = self.execute()
        return result

    def execute(self):
        generator = outlines.Generator(model, Decision)
        result = generator(self.prompt, max_tokens=1024, temperature=0.5, seed=42)
        return result


def query(question, max_turns=5):
    i = 0
    next_prompt = (
        "\n<|im_start|>user\n" + question + "<|im_end|>" "\n<|im_start|>assistant\n"
    )
    previous_actions = []
    while i < max_turns:
        i += 1
        # prompt = hermes_prompt(
        #     # question=question,
        #     schema=Decision.model_json_schema(),
        #     # today=datetime.datetime.today().strftime("%Y-%m-%d"),
        # )
        prompt = hermes_prompt()
        bot = ChatBot(prompt=prompt)
        result = bot(next_prompt)
        json_result = json.loads(result)["Decision"]
        if "Final_Answer" not in list(json_result.keys()):
            scratchpad = json_result["Scratchpad"] if i == 0 else ""
            thought = json_result["Thought"]
            print(f"\x1b[34m Scratchpad: {scratchpad} \x1b[0m")
            print(f"\x1b[34m Thought: {thought} \x1b[0m")
            next_prompt += "\nScratchpad: " + scratchpad + "\nThought: " + thought
        else:
            scratchpad = json_result["Scratchpad"]
            final_answer = json_result["Final_Answer"]
            print(f"\x1b[34m Scratchpad: {scratchpad} \x1b[0m")
            print(f"\x1b[34m Final Answer: {final_answer} \x1b[0m")
            return final_answer
    print(
        f"\nFinal Answer: I am sorry, but I am unable to answer your question. Please provide more information or a different question."
    )
    return "No answer found"


question = """
reminging length,is paper replacement
100, False
83, False
61, False
40, False
29, False
18, False
1, False
0, True
100, False
56, False
19, False
2, False
200, False
180, False
130, False
100, False
100, False
100, False
100, False

try to diagnose the log.
"""


# print(query("What's 2 to the power of 10?"))
print(query(question))
