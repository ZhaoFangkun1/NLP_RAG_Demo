# -*- coding: UTF-8 -*-
"""
@File    ：metric.py
@Author  ：zfk
@Date    ：2024/5/9 16:15
"""
import json

import nltk
from rouge import Rouge
from bert_score import score


def metric(pred: str, answer_list: list[str]):
    # 计算BLEU
    print(pred, answer_list)
    reference = [answer.split() for answer in answer_list]
    candidate = pred.split()
    bleu_score = nltk.translate.bleu_score.sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    # 计算ROUGE
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred, ' '.join(answer_list))
    
     # 确保pred和每个答案都是列表形式
    preds = [pred.lower().strip()]
    refs = [answer.lower().strip() for answer in answer_list]
    
    # 计算BERTScore的精确度、召回率和F1分数
    bs = score(preds, refs, model_type='bert-base-uncased', num_layers=12, idf=False, device= 'cuda')
    
    # BERTScore返回的是一个包含三个元素的元组，分别对应精确度、召回率和F1分数
    bert_p, bert_r, bert_f = bs

    return bleu_score, rouge_scores[0]['rouge-l'], bert_p, bert_r, bert_f


if __name__ == '__main__':
    with open('../data/crag_200_result.jsonl', 'r') as f:
        lines = f.readlines()
    bleu_scores = []
    bert_p_scores = []
    bert_r_scores = []
    bert_f_scores = []
    rouge_l_p = []
    rouge_l_r = []
    rouge_l_f = []
    for line in lines:
        data = json.loads(line)
        query = data['query']
        answer = data['answer']
        pred = data['pred']
        bleu, rouge_l, bert_p, bert_r, bert_f = metric(pred.lower().strip(), [answer.lower().strip()])
        bleu_scores.append(bleu)
        rouge_l_p.append(rouge_l['p'])
        rouge_l_r.append(rouge_l['r'])
        rouge_l_f.append(rouge_l['f'])
        bert_p_scores.append(bert_p)
        bert_r_scores.append(bert_r)
        bert_f_scores.append(bert_f)

    print(f"BLEU: {sum(bleu_scores) / len(bleu_scores)}")
    print(f"ROUGE-L P: {sum(rouge_l_p) / len(rouge_l_p)}")
    print(f"ROUGE-L R: {sum(rouge_l_r) / len(rouge_l_r)}")
    print(f"ROUGE-L F: {sum(rouge_l_f) / len(rouge_l_f)}")
    print(f"BERTScore P: {sum(bert_p_scores) / len(bert_p_scores)}")
    print(f"BERTScore R: {sum(bert_r_scores) / len(bert_r_scores)}")
    print(f"BERTScore F: {sum(bert_f_scores) / len(bert_f_scores)}")

