# -*- coding: UTF-8 -*-
"""
@Project ：rag 
@File    ：Basic_RAG_pipeline.py
@Author  ：zfk
@Date    ：2024/5/7 22:21
"""
import torch
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from model_response import MyApiLLM, MyLocalLLM


def main(args):
    # 1. 从目录加载文档，一个txt为一个文档
    documents = SimpleDirectoryReader(input_files=[args.data_path]).load_data()
    # 2.定义解析器
    node_parser = SimpleNodeParser.from_defaults(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    # 3.将文档解析为TextNode
    nodes = node_parser.get_nodes_from_documents(documents)
    # 4.将TextNode转换为索引向量
    index = VectorStoreIndex(nodes)
    # 5.创建查询引擎
    query_engine = index.as_query_engine()
    # 6.查询
    response = query_engine.query(
        "When did Musk establish xAI"
    )
    print(str(response))
    print(response.source_nodes[0].text)
    print(response.source_nodes[1].text)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='api', choices=['api', 'local'])

    parser.add_argument('--api_key', type=str, help='api_key', default='')
    parser.add_argument('--secret_key', type=str, help='secret_key', default='')

    parser.add_argument('--llm_model_path', type=str, help='local llm model path', default='../qwen1.5-0.5B')

    parser.add_argument('--embedding_model_path', type=str, help='local embedding model path',
                        default='../BAAI/bge-small-en-v1.5')

    parser.add_argument('--data_path', type=str, help='local data path', default='../data/Elon.txt')
    parser.add_argument('--chunk_size', type=int, default=64, help='chunk size')
    parser.add_argument('--chunk_overlap', type=int, default=2, help='chunk overlap')
    args = parser.parse_args()
    if args.model_type == 'api':
        assert args.api_key and args.secret_key, "api_key and secret_key must be provided"
        llm = MyApiLLM(args.api_key, args.secret_key)
    else:
        llm = MyLocalLLM(args.llm_model_path)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_path)
    main(args)
