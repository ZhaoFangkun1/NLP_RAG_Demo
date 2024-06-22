import json
import re

from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

source_file = ''
target_file = ''


def remove_special_tokens(text):
    # 去除特殊token的正则表达式
    special_tokens = re.compile(r'\[gMASK]|\bsop\b')
    return special_tokens.sub('', text).strip()


class LocalPeftModel:
    def __init__(self, peft_model_path):
        self.model = AutoPeftModelForCausalLM.from_pretrained(peft_model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)
        self.model = self.model.to("cuda").eval()

    def chat(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()
        outputs = self.model.generate(inputs, max_new_tokens=256)
        response = self.tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True)
        response = remove_special_tokens(response)
        return response


def main():
    my_local_llm = LocalPeftModel("/PATH/TO/LORA_WEIGHT")

    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    result = []
    for line in tqdm(lines, desc="Generating responses"):
        data = json.loads(line)

        pred = my_local_llm.complete(data['query'])

        result.append(json.dumps({'query': data["query"], 'answer': data["answer"], 'pred': pred.text},
                                 ensure_ascii=False) + '\n')

    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(''.join(result))


if __name__ == '__main__':
    main()
