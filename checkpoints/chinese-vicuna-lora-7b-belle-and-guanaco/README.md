---
license: gpl-3.0
datasets:
- BelleGroup/generated_train_0.5M_CN
- JosephusCheung/GuanacoDataset
language:
- zh
tags:
- alpaca
- Chinese-Vicuna
- llama
---

This is a Chinese instruction-tuning lora checkpoint based on llama-7B from [this repo's](https://github.com/Facico/Chinese-Vicuna) work

You can use it like this: 



```python
from transformers import LlamaForCausalLM
from peft import PeftModel

model = LlamaForCausalLM.from_pretrained(
    "decapoda-research/llama-7b-hf",
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    LORA_PATH, # specific checkpoint path from "Chinese-Vicuna/Chinese-Vicuna-lora-7b-belle-and-guanaco"
    torch_dtype=torch.float16,
    device_map={'': 0}
)
```