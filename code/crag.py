# -*- coding: UTF-8 -*-
"""
@File    ：crag.py
@Author  ：zfk
@Date    ：2024/5/9 12:55
"""
import json
import os
from llama_index.core import Document,Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from model_response import MyApiLLM, MyLocalLLM


def build_automerging_index(documents, save_dir="merging_index", chunk_sizes=None):
    if not os.path.exists(save_dir):
        print('creating index directory', save_dir)
        chunk_sizes = chunk_sizes or [2048, 512, 128]
        node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=chunk_sizes)
        nodes = node_parser.get_nodes_from_documents(documents)
        leaf_nodes = get_leaf_nodes(nodes)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)
        automerging_index.storage_context.persist(persist_dir=save_dir)
    else:
        print('loading index directory', save_dir)
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=save_dir))
    return automerging_index


def test_crag(source_file='../data/crag_data_200.jsonl', target_file='../data/crag_200_result.jsonl'):
    with open(source_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    result = []
    for i,line in enumerate(lines[:5]):
        data = json.loads(line)
        query = data['query']
        answer = data['answer']
        search_results = data['search_results']
        documents = []
        for search_result in search_results:
            documents.append(Document(text=search_result['page_result']))
        index = build_automerging_index(documents, save_dir='merging_index/{}.index'.format(i))
        base_retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)
        retriever = AutoMergingRetriever(base_retriever, index.storage_context, verbose=True)
        auto_merging_engine = RetrieverQueryEngine.from_args(retriever, node_postprocessors=[Settings.rerank_model])
        auto_merging_response = auto_merging_engine.query(query)
        print(f'{query=}')
        print(f'{auto_merging_response.response=}')
        print(f'{answer=}')
        data['pred'] = auto_merging_response.response
        result.append(json.dumps({'query': query, 'answer': answer, 'pred': auto_merging_response.response},ensure_ascii=False)+'\n')
    with open(target_file, 'w', encoding='utf-8') as f:
        f.write(''.join(result))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='api', choices=['api', 'local'])
    parser.add_argument('--api_key', type=str, help='api_key', default='')
    parser.add_argument('--secret_key', type=str, help='secret_key', default='')
    parser.add_argument('--llm_model_path', type=str, help='local llm model path', default='../qwen1.5-0.5B')
    parser.add_argument('--embedding_model_path', type=str, help='local embedding model path',
                        default='../BAAI/bge-small-en-v1.5')
    parser.add_argument('--similarity_top_k', type=int, default=12)
    parser.add_argument('--data_path', type=str, help='local data path', default='../data/Henry.txt')
    parser.add_argument('--save_path', type=str, help='chunk save path', default='./merging_index')
    parser.add_argument('--rerank_model_path', type=str, help='local rerank model path',
                        default='../BAAI/bge-reranker-base')
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
    test_crag()
