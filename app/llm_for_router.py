import outlines
import llama_cpp

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
gen = outlines.generate.choice(model, ["Blue", "Red", "Yellow"])

color = gen("What is the closest color to Indigo? ")
print(color)
