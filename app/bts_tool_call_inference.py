import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from unsloth import FastLanguageModel
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import re


class LogArgs(BaseModel):
    start_time: str = Field(
        description="The start time of the log query range in 'YYYY-MM-DD HH:MM:SS.sss' format (e.g., '2026-01-08 14:03:50.690')."
    )
    end_time: str = Field(
        description="The end time of the log query range in 'YYYY-MM-DD HH:MM:SS.sss' format (e.g., '2026-01-08 15:03:50.690')."
    )
    desire_material: str = Field(
        "The desire material from user(e.g., P.-.-.8.J or N.-.-.7.N)"
    )


class TrackArgs(BaseModel):
    start_time: str = Field(
        description="The start time of the log query range in 'YYYY-MM-DD HH:MM:SS.sss' format (e.g., '2026-01-08 14:03:50.690')."
    )
    end_time: str = Field(
        description="The end time of the log query range in 'YYYY-MM-DD HH:MM:SS.sss' format (e.g., '2026-01-08 15:03:50.690')."
    )
    material: str = Field(description="The material is formatted like N.-.-.7.N")


@tool(args_schema=LogArgs)
def get_material_change_in_log(start_time: str, end_time: str):
    """get material chaneg in log"""
    return "get_material_change_in_log"


@tool(args_schema=LogArgs)
def get_glue_set_func_call_in_log(start_time: str, end_time: str, desire_material: str):
    """get the glue set func call in log, this func will extract the set func event with material lifecycle events"""
    return "## get_glue_set_func_call_in_log \n\n ### hello glue set func \n\n asdadadasdasdasd"


@tool(args_schema=TrackArgs)
def track_material_in_log(start_time: str, end_time: str, material: str):
    """track the material in log, to show the material lifecycle"""
    return "track_material_in_log"


tools = {
    "get_material_change_in_log": get_material_change_in_log,
    "get_glue_set_func_call_in_log": get_glue_set_func_call_in_log,
    "track_material_in_log": track_material_in_log,
}

model_path = "D:/code/gguf-models/bts-tool-call-functiongemma-270m-transformers-default-v1/functiongemma_lora"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path
    # max_seq_length=2048,
    # dtype=None,
    # load_in_4bit=False,
    # local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Move model to CPU
device = torch.device("cpu")
model.to(device)

TOOLS = list(tools.values())


def extract_tool_calls(text):
    def cast(v):
        try:
            return int(v)
        except:
            try:
                return float(v)
            except:
                return {"true": True, "false": False}.get(v.lower(), v.strip("'\""))

    return [
        {
            "name": name,
            "arguments": {
                k: cast((v1 or v2).strip())
                for k, v1, v2 in re.findall(
                    r"(\w+):(?:<escape>(.*?)<escape>|([^,}]*))", args
                )
            },
        }
        for name, args in re.findall(
            r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>",
            text,
            re.DOTALL,
        )
    ]


def process_tool_calls(output, messages):
    calls = extract_tool_calls(output)
    if not calls:
        return messages
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": call} for call in calls],
        }
    )
    results = [
        {"name": c["name"], "response": tools[c["name"]](**c["arguments"])}
        for c in calls
    ]
    messages.append({"role": "tool", "content": results})
    return messages


def _do_inference(model, messages, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    output = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)

    out = model.generate(
        **inputs.to(model.device),
        max_new_tokens=max_new_tokens,
        top_p=0.95,
        top_k=64,
        temperature=1.0,
    )
    generated_tokens = out[0][len(inputs["input_ids"][0]) :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


def do_inference(model, messages, print_assistant=True, max_new_tokens=128):
    output = _do_inference(model, messages, max_new_tokens=max_new_tokens)
    messages = process_tool_calls(output, messages)
    if messages[-1]["role"] == "tool":
        output = _do_inference(model, messages, max_new_tokens=max_new_tokens)
    messages.append({"role": "assistant", "content": output})
    if print_assistant:
        print(output)
    return messages


messages = []
messages.append({"role": "user", "content": "查看一下材质N.-.-.8.J"})
messages = do_inference(model, messages, max_new_tokens=128)
print(messages)
print(messages[-1])
