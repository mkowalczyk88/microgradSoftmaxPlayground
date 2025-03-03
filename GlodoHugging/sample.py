import sys
import importlib.util
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import set_seed
from peft import PeftModel

set_seed(42)

def import_config(filename):
    if not os.path.exists(filename):
        print(f"Config file '{filename}' not found.")
        return

    module_name = os.path.splitext(os.path.basename(filename))[0]
    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config = import_config(sys.argv[1] if len(sys.argv) > 1 else "configs/default.py")

device = "cuda"
#model_path = "sdadas/polish-gpt2-small"
#model_path = "out/fine_tuned_model-large"
model_path = "out/lora-fine_tuned_model-large"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(config.model_data, device_map="auto")

model = PeftModel.from_pretrained(model, model_path)
model.eval()
#model.compile()


inputs = tokenizer("Głodomorek i marchewka", return_tensors="pt").to(device)
input_ids = inputs.input_ids
attention_mask = inputs["attention_mask"]

outputs = model.generate(inputs['input_ids'],
                           attention_mask=attention_mask,
                           pad_token_id=tokenizer.eos_token_id,
                           #pad_token_id=None,
                           #max_new_tokens=100,
                           max_length=200,
                           temperature=0.8,
                           top_k=200,
                           num_return_sequences=2,
                           do_sample=True,
                         )
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
print("-------------------")
print(tokenizer.decode(outputs[1], skip_special_tokens=False))
