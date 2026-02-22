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
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
        "NousResearch/Hermes-2-Pro-Llama-3-8B"
    ),
    n_gpu_layers=-1,
    flash_attn=True,
    n_ctx=8192,
    verbose=False,
)
model = outlines.from_llamacpp(llm)


def wikipedia(q):
    # return httpx.get(
    #     "https://en.wikipedia.org/w/api.php",
    #     params={"action": "query", "list": "search", "srsearch": q, "format": "json"},
    # ).json()["query"]["search"][0]["snippet"]
    return httpx.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": q, "format": "json"},
    )


def calculate(numexp):
    return eval(numexp)


class Action(str, Enum):
    wikipedia = "wikipedia"
    calculate = "calculate"


class Reason_and_Act(BaseModel):
    Scratchpad: str = Field(
        ...,
        description="Information from the Observation useful to answer the question",
    )
    Thought: str = Field(
        ...,
        description="It describes your thoughts about the question you have been asked",
    )
    Action: Action
    Action_Input: str = Field(..., description="The arguments of the Action.")


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
        result = generator(self.prompt, max_tokens=1024, temperature=0, seed=42)
        return result


def query(question, max_turns=5):
    i = 0
    next_prompt = (
        "\n<|im_start|>user\n" + question + "<|im_end|>" "\n<|im_start|>assistant\n"
    )
    previous_actions = []
    while i < max_turns:
        i += 1
        prompt = hermes_prompt(
            # question=question,
            schema=Decision.model_json_schema(),
            # today=datetime.datetime.today().strftime("%Y-%m-%d"),
        )
        bot = ChatBot(prompt=prompt)
        result = bot(next_prompt)
        json_result = json.loads(result)["Decision"]
        if "Final_Answer" not in list(json_result.keys()):
            scratchpad = json_result["Scratchpad"] if i == 0 else ""
            thought = json_result["Thought"]
            action = json_result["Action"]
            action_input = json_result["Action_Input"]
            print(f"\x1b[34m Scratchpad: {scratchpad} \x1b[0m")
            print(f"\x1b[34m Thought: {thought} \x1b[0m")
            print(f"\x1b[36m  -- running {action}: {str(action_input)}\x1b[0m")
            if action + ": " + str(action_input) in previous_actions:
                observation = (
                    "You already run that action. **TRY A DIFFERENT ACTION INPUT.**"
                )
            else:
                if action == "calculate":
                    try:
                        observation = eval(str(action_input))
                    except Exception as e:
                        observation = f"{e}"
                elif action == "wikipedia":
                    try:
                        observation = wikipedia(str(action_input))
                    except Exception as e:
                        observation = f"{e}"
            print()
            print(f"\x1b[33m Observation: {observation} \x1b[0m")
            print()
            previous_actions.append(action + ": " + str(action_input))
            next_prompt += (
                "\nScratchpad: "
                + scratchpad
                + "\nThought: "
                + thought
                + "\nAction: "
                + action
                + "\nAction Input: "
                + action_input
                + "\nObservation: "
                + str(observation)
            )
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


# print(query("What's 2 to the power of 10?"))
print(
    query(
        "What does England share borders with? You need to search the answer using wikipedia tools."
    )
)
