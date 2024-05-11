# -*- coding: UTF-8 -*-
"""
@Project ：rag 
@File    ：Sentence_window_retrieval.py
@Author  ：zfk
@Date    ：2024/5/7 22:44
"""
import os
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from model_response import MyApiLLM, MyLocalLLM


def build_sentence_window_index(documents, sentence_window_size=3, save_dir="sentence_index"):
    # 1. create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    Settings.node_parser = node_parser
    # 2. create or load the sentence index
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(documents)
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
    return sentence_index


def main(args):
    # 1. 从目录加载文档，一个txt为一个文档
    documents = SimpleDirectoryReader(input_files=[args.data_path]).load_data()
    # 2.所有文档合并成一个Doc对象
    document = Document(text="\n\n".join([doc.text for doc in documents]))
    # 3.构建索引
    index = build_sentence_window_index([document], save_dir=args.save_path)
    # 4.后处理文档
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    # 5. create the query engine
    sentence_window_engine = index.as_query_engine(
        similarity_top_k=args.similarity_top_k, node_postprocessors=[postproc, Settings.rerank_model]
    )
    # 6.查询
    window_response = sentence_window_engine.query("When did Musk establish xAI")
    # 7.打印结果
    print(window_response)
    print('----------')
    print(window_response.source_nodes[0].get_text())
    print('----------')
    print(window_response.source_nodes[1].get_text())


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='api', choices=['api', 'local'])
    parser.add_argument('--api_key', type=str, help='api_key', default='')
    parser.add_argument('--secret_key', type=str, help='secret_key', default='')
    parser.add_argument('--llm_model_path', type=str, help='local llm model path', default='../qwen1.5-0.5B')
    parser.add_argument('--embedding_model_path', type=str, help='local embedding model path',default='../BAAI/bge-small-en-v1.5')
    parser.add_argument('--similarity_top_k', type=int, default=12)
    parser.add_argument('--data_path', type=str, help='local data path', default='../data/Elon.txt')
    parser.add_argument('--save_path', type=str, help='chunk save path', default='./sentence_index')
    parser.add_argument('--rerank_model_path', type=str, help='local rerank model path', default='../BAAI/bge-reranker-base')
    parser.add_argument('--rerank_top_n', type=int, default=2)
    args = parser.parse_args()

    if args.model_type == 'api':
        assert args.api_key and args.secret_key, "api_key and secret_key must be provided"
        llm = MyApiLLM(args.api_key, args.secret_key)
    else:
        llm = MyLocalLLM(args.llm_model_path)
    Settings.llm = llm
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_path)
    Settings.rerank_model = SentenceTransformerRerank(top_n=args.rerank_top_n, model=args.rerank_model_path)
    main(args)
