# -*- coding: UTF-8 -*-
"""
@Project ：rag 
@File    ：model_response.py
@Author  ：zfk
@Date    ：2024/5/8 11:45
"""
import requests
import json
from typing import Any
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import AutoTokenizer, AutoModelForCausalLM


class ApiModel:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.access_token = self.get_access_token()
        self.Yi_34b_chat_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/yi_34b_chat?access_token=" + self.access_token

    def get_access_token(self):
        """
        使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
        """
        url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={self.api_key}&client_secret={self.secret_key}"
        payload = json.dumps("")
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.json().get("access_token")

    def chat(self, prompt, top_p=0.5, top_k=5, temperature=1.0, penalty_score=1, ):
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            'top_p': 0.5, 'top_k': 5, 'temperature': 1.0, 'penalty_score': 1
        })
        headers = {'Content-Type': 'application/json'}
        response = requests.request("POST", self.Yi_34b_chat_url, headers=headers, data=payload)
        print(response.json())
        return response.json()['result']


class LocalModel:
    def __init__(self, pretrained_model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True).eval()
        self.model = self.model.float()

    def chat(self, prompt):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()  # GPU方式
        outputs = self.model.generate(inputs, max_length=self.num_output)
        response = self.tokenizer.decode(outputs[0])
        return response


class MyLocalLLM(CustomLLM):
    context_window = 2048
    num_output = 256
    model = "xxxx"
    model_name = "xxxx"

    def __init__(self, pretrained_model_name_or_path):
        super().__init__()
        self.model = LocalModel(pretrained_model_name_or_path)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        # 完成函数
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()
        outputs = self.model.generate(inputs, max_length=self.num_output)
        response = self.tokenizer.decode(outputs[0])
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        # 流式完成函数
        print("流式完成函数")
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').cuda()  # GPU方式
        outputs = self.model.generate(inputs, max_length=self.num_output)
        response = self.tokenizer.decode(outputs[0])
        for token in response:
            yield CompletionResponse(text=token, delta=token)


class MyApiLLM(CustomLLM):
    context_window = 8192
    num_output = 256
    model = "Yi_34b_chat"
    model_name = "Yi_34b_chat"

    def __init__(self, api_key, secret_key):
        super().__init__()
        self.model = ApiModel(api_key, secret_key)

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # 得到LLM的元数据
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()  # 回调函数
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:

        response = self.model.chat(prompt)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        # 流式完成函数
        response = self.model.chat(prompt)
        for token in response:
            yield CompletionResponse(text=token, delta=token)


if __name__ == '__main__':
    api_key = ""
    secret_key = ""
    my_api_llm = MyApiLLM(api_key, secret_key)
    print(my_api_llm.complete('你好'))
