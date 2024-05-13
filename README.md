# Retrieval Augmented Generation (RAG) Intro Project 🤖🔍📝

**RAG教学示例项目**  
1. 项目使用了[LlamaIndex](https://github.com/run-llama/llama_index)框架演示了RAG的基本流程，也欢迎使用其他框架如[LangChain](https://www.langchain.com/)等进行实验。
2. 项目主体来自[Retrieval-Augmented-Generation-Intro-Project](https://github.com/HenryHengLUO/Retrieval-Augmented-Generation-Intro-Project/blob/main/README.md)，本项目对llama_index的最新版本进行了适配。
3. 项目使用了[百度千帆](https://console.bce.baidu.com/qianfan/overview)提供的Yi-34B-Chat的接口(限时免费)，需自行申请API Key和Secret Key。也可尝试使用其它免费的API如[阿里云](https://help.aliyun.com/zh/dashscope/developer-reference/?spm=a2c4g.11186623.0.0.644e9b6em7thMV)，需在代码中相应位置做替换。

## Project Structure 📂

- `README.md`: 项目总览

- **code**:  文件夹下包含三个教学实验，三个实验均有各自对应.ipynb,.py以及.sh可直接运行；同时提供了在[KDD CUP 2024 CRAG: Comprehensive Rag Benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)的训练集上的测试以及结果评测脚本。
  - `1_Basic_RAG_Pipeline`: RAG基础的pipeline演示 
  ![image](data/llamaindex_rag_overview.png)
  - `2_Sentence_window_retrieval`: 检索时使用小的chunk，将检索结果的上下文都拼接到prompt中

    <img src="data/llamaindex_SentenceWindowRetrieval_overview.png" width="400" height="300"><img src="data/llamaindex_SentenceWindowRetrieval_example.png" width="400" height="300">

  - `3_Auto-merging_Retrieval`: 将文档按照块大小拆分成不同层级的节点，在检索时使用叶子节点检索，然后检查父节点包含的子节点中被检索到的比例，高于一定阈值时将父节点作为检索结果，否则被检索到的子节点作为检索结果，AutoMergingRetrieval的详细介绍可参考 https://zhaozhiming.github.io/2024/03/19/auto-merging-rag/
  ![image](data/llamaindex_AutoMergingRetrieval_example.png)
  - `model_response.py`: 提供了API访问和本地部署LLM两种方式，选择本地部署的同学可以进一步改造代码，使用`vllm`框架加速推理
  - `crag.sh`: [KDD CUP 2024 CRAG:Comprehensive Rag Benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)训练集的测试脚本
  - `metric.py`: 评测脚本，计算模型生成内容与标答的BLEU和rouge-l指标
- **data**: 实验所需的语料，包括
  - `Elon.txt`: 示例文件 `Elon.txt`
  - [KDD CUP 2024 CRAG:Comprehensive Rag Benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)数据，每条数据包含**query**、**可能和query相关的五个网页内容**(为方便使用已经进行了简单的html标签去除)、**query对应的答案**。提供CRAG全量训练集2735条数据和200条测试数据，如有训练需求可自行将CRAG全量训练集切分作为你自己的训练集和验证集。

## Getting Started 🚀

1. 克隆或下载项目仓库到本地：
```shell
  git clone https://github.com/ZhaoFangkun1/NLP_RAG_Demo.git
```
2. 准备环境
```shell
  conda create -n rag python=3.10
  conda activate rag
  pip install llama_index
  pip install llama-index-embeddings-huggingface
```
3. 进入`code`文件夹，依次运行三种实验脚本
```shell
cd code
1. sh Basic_RAG_Pipeline.sh
2. sh Sentence_window_retrieval.sh
3. sh Auto-merging_Retrieval.sh
```

4. 更改数据为提供的[CRAG](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)数据集进行测试，同时也可尝试其他的数据集
```shell
sh crag.sh
python metric.py
```
5. 鼓励进行模型训练或流程上的改进

## 项目要求：下面三项内容，至少选择其二开展进一步探索并完成实验报告
1. 检索策略优化：尝试更多的chunk策略，以及更好的检索召回和重排算法
2. 微调BGE的检索和重排模型：
   - 自行将CRAG全量训练集进行划分，微调BGE检索以及重排模型，可参考 https://github.com/FlagOpen/FlagEmbedding/blob/master/examples/finetune/README.md
   - 微调时需要构造query对应的正负例，但是CRAG数据集并未给出query对应的chunk正例（只给出了与每个query可能相关的网页、每个query对应的答案），因此需要设计方案为query构造正例（除人工标注外，还可通过prompt LLM进行query和chunk的相关性判定）
3. 对LLM进行进一步预训练或SFT：
   - 根据自己的算力情况，使用给定的CRAG数据对LLM做进一步的预训练或（和）SFT
   - 对使用的LLM框架不做限制，可参考[llama-factory](https://github.com/hiyouga/LLaMA-Factory),[megatron-lm](https://github.com/NVIDIA/Megatron-LM)框架，以及阿里进行二次封装之后的[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)框架
