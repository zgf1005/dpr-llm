'''
@Description: 
@Author: shenlei
@Date: 2023-11-28 14:04:27
@LastEditTime: 2024-05-13 17:04:41
@LastEditors: shenlei
'''
import logging
import torch

import numpy as np

from tqdm import tqdm
from typing import List, Dict, Tuple, Type, Union
from copy import deepcopy

#from .utils import reranker_tokenize_preproc

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from BCEmbedding.utils import logger_wrapper
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logger = logger_wrapper('BCEmbedding.models.RerankerModel')

def reranker_tokenize_preproc(
    query: str, 
    passages: List[str],
    tokenizer=None,
    max_length: int=512,
    overlap_tokens: int=80,
    ):
    assert tokenizer is not None, "Please provide a valid tokenizer for tokenization!"
    sep_id = tokenizer.sep_token_id

    def _merge_inputs(chunk1_raw, chunk2):
        chunk1 = deepcopy(chunk1_raw)

        chunk1['input_ids'].append(sep_id)
        chunk1['input_ids'].extend(chunk2['input_ids'])
        chunk1['input_ids'].append(sep_id)

        chunk1['attention_mask'].append(chunk2['attention_mask'][0])
        chunk1['attention_mask'].extend(chunk2['attention_mask'])
        chunk1['attention_mask'].append(chunk2['attention_mask'][0])

        if 'token_type_ids' in chunk1:
            token_type_ids = [1 for _ in range(len(chunk2['token_type_ids'])+2)]
            chunk1['token_type_ids'].extend(token_type_ids)
        return chunk1
   

 
    if len(query) > 400:
        query = query[:400] 
    query_inputs = tokenizer.encode_plus(query, truncation=False, padding=False)
    max_passage_inputs_length = max_length - len(query_inputs['input_ids']) - 2
    assert max_passage_inputs_length > 100, "Your query is too long! Please make sure your query less than 400 tokens!"
    overlap_tokens_implt = min(overlap_tokens, max_passage_inputs_length//4)
    
    res_merge_inputs = []
    res_merge_inputs_pids = []
    for pid, passage in enumerate(passages):
        passage_inputs = tokenizer.encode_plus(passage, truncation=False, padding=False, add_special_tokens=False)
        passage_inputs_length = len(passage_inputs['input_ids'])

        if passage_inputs_length <= max_passage_inputs_length:
            qp_merge_inputs = _merge_inputs(query_inputs, passage_inputs)
            res_merge_inputs.append(qp_merge_inputs)
            res_merge_inputs_pids.append(pid)
        else:
            start_id = 0
            while start_id < passage_inputs_length:
                end_id = start_id + max_passage_inputs_length
                sub_passage_inputs = {k:v[start_id:end_id] for k,v in passage_inputs.items()}
                start_id = end_id - overlap_tokens_implt if end_id < passage_inputs_length else end_id

                qp_merge_inputs = _merge_inputs(query_inputs, sub_passage_inputs)
                res_merge_inputs.append(qp_merge_inputs)
                res_merge_inputs_pids.append(pid)
    
    return res_merge_inputs, res_merge_inputs_pids
class RerankerModel:
    def __init__(
            self,
            model_name_or_path: str='/root/shared-nvme/bce-reranker-base_v1',
            use_fp16: bool=False,
            device: str=None,
            **kwargs
        ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **kwargs)
        logger.info(f"Loading from `{model_name_or_path}`.")
        
        num_gpus = torch.cuda.device_count()
        if device is None:
            self.device = "cuda" if num_gpus > 0 else "cpu"
        else:
            self.device = 'cuda:{}'.format(int(device)) if device.isdigit() else device
        
        if self.device == "cpu":
            self.num_gpus = 0
        elif self.device.startswith('cuda:') and num_gpus > 0:
            self.num_gpus = 1
        elif self.device == "cuda":
            self.num_gpus = num_gpus
        else:
            raise ValueError("Please input valid device: 'cpu', 'cuda', 'cuda:0', '0' !")

        if use_fp16:
            self.model.half()

        self.model.eval()
        self.model = self.model.to(self.device)

        if self.num_gpus > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        logger.info(f"Execute device: {self.device};\t gpu num: {self.num_gpus};\t use fp16: {use_fp16}")

        # for advanced preproc of tokenization
        self.max_length = kwargs.get('max_length', 512)
        self.overlap_tokens = kwargs.get('overlap_tokens', 80)
    
    def compute_score(
            self, 
            sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], 
            batch_size: int = 256,
            max_length: int = 512,
            enable_tqdm: bool=True,
            **kwargs
        ):
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus
        
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        with torch.no_grad():
            scores_collection = []
            for sentence_id in tqdm(range(0, len(sentence_pairs), batch_size), desc='Calculate scores', disable=not enable_tqdm):
                sentence_pairs_batch = sentence_pairs[sentence_id:sentence_id+batch_size]
                inputs = self.tokenizer(
                            sentence_pairs_batch, 
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                inputs_on_device = {k: v.to(self.device) for k, v in inputs.items()}
                scores = self.model(**inputs_on_device, return_dict=True).logits.view(-1,).float()
                scores = torch.sigmoid(scores)
                scores_collection.extend(scores.cpu().numpy().tolist())
        
        if len(scores_collection) == 1:
            return scores_collection[0]
        return scores_collection

    def rerank(
            self,
            query: str,
            passages: List[str],
            batch_size: int=256,
            **kwargs
        ):
        # remove invalid passages
        passages = [p[:128000] for p in passages if isinstance(p, str) and 0 < len(p)]
        if query is None or len(query) == 0 or len(passages) == 0:
            return {'rerank_passages': [], 'rerank_scores': []}
        
        # preproc of tokenization
        sentence_pairs, sentence_pairs_pids = reranker_tokenize_preproc(
            query, passages, 
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            overlap_tokens=self.overlap_tokens,
            )

        # batch inference
        if self.num_gpus > 1:
            batch_size = batch_size * self.num_gpus

        tot_scores = []
        with torch.no_grad():
            for k in range(0, len(sentence_pairs), batch_size):
                batch = self.tokenizer.pad(
                        sentence_pairs[k:k+batch_size],
                        padding=True,
                        max_length=None,
                        pad_to_multiple_of=None,
                        return_tensors="pt"
                    )
                batch_on_device = {k: v.to(self.device) for k, v in batch.items()}
                scores = self.model(**batch_on_device, return_dict=True).logits.view(-1,).float()
                scores = torch.sigmoid(scores)
                tot_scores.extend(scores.cpu().numpy().tolist())

        # ranking
        merge_scores = [0 for _ in range(len(passages))]
        for pid, score in zip(sentence_pairs_pids, tot_scores):
            merge_scores[pid] = max(merge_scores[pid], score)

        merge_scores_argsort = np.argsort(merge_scores)[::-1]
        sorted_passages = []
        sorted_scores = []
        for mid in merge_scores_argsort:
            sorted_scores.append(merge_scores[mid])
            sorted_passages.append(passages[mid])
        
        return {
            'rerank_passages': sorted_passages,
            'rerank_scores': sorted_scores,
            'rerank_ids': merge_scores_argsort.tolist()
        }
import json

re=RerankerModel()
""" with open("/root/shared-nvme/DPR/fl-xr_dpr_dpr/xr-dpr-20.json","r") as f:
    test_data=json.load(f)
q=[]
passages=[]
for i in range(len(test_data)):
    s=""
    for j in range(len(test_data[i]["answers"])):
        s+=test_data[i]["answers"][j]
    q.append(s)
    ans=[]
    for j in range(len(test_data[i]["ctxs"])):
        ans.append(test_data[i]["ctxs"][j]["text"])
    passages.append(ans)  
reranker=[]
for i in range(len(passages)):
    reranker.append(re.rerank(q[i],passages[i],256)) 
    print(i)
a=[]
for i in range(len(reranker)):
    a.append({
        "ques_title":test_data[i]['question'],
        "ans_contents":[reranker[i]['rerank_passages'][0],reranker[i]['rerank_passages'][1],reranker[i]['rerank_passages'][2]]
    }) 
with open("/root/shared-nvme/DPR/fl-xr_dpr_dpr/xr-dpr-20-reranker.json","w") as f:
    json.dump(a,f,ensure_ascii=False) 
print(1)  """
#test= [json.loads(line) for line in open("/root/shared-nvme/loc14_test.json","r")]
with open("/root/shared-nvme/reference-fl/001_reference.txt","r") as f:
    test_data_ture=f.readlines()
with open("/root/shared-nvme/bm25-doc-bge/fl-xr-es-250227-top20.json","r") as f:
    test_data_es=json.load(f)
q = []
passages=[]
for i in range(len(test_data_es)):
    s=test_data_ture[i]
    s+="["
    s+=" ".join(test_data_es[i]["categories"])
    s+="]"
    q.append(s)
    ans=[]
    for j in range(len(test_data_es[i]["ralation"])):
        ss=""
        for k in range(len(test_data_es[i]["ralation"][j]["ans_contents"])):
            ss+=test_data_es[i]["ralation"][j]["ans_contents"][k]
        ss+="["
        ss+=" ".join(test_data_es[i]["ralation"][j]["categories"])
        ss+="]"
        ans.append(ss)
    passages.append(ans) 
 

reranker = []
for i in range(len(q)):
    reranker.append(re.rerank(q[i],passages[i],256)) 
    print(i) 

""" ans=0
a=[]
for i in range(len(reranker)):
    key=""
    for j in test_data[i]["categories"]:
        key+=j
    qq=q[i].replace(" ", "")
    try:
        index = reranker[i]["rerank_passages"].index(qq)  # 查找 a 在 b 中的索引
        ans+=1/(index+1)
    except ValueError:
        print(0)
print(ans/5000) """
import re
a=[]
for i in range(len(reranker)):
    key="["
    key+=" ".join(test_data_es[i]["categories"])
    key+=']'
    reranker[i]['rerank_passages'][0]=re.sub(r'\[.*?\]', '', reranker[i]['rerank_passages'][0])
    reranker[i]['rerank_passages'][1]=re.sub(r'\[.*?\]', '', reranker[i]['rerank_passages'][1])
    reranker[i]['rerank_passages'][2]=re.sub(r'\[.*?\]', '', reranker[i]['rerank_passages'][2])
    a.append({
        "ques_title":test_data_ture[i],
        "ans_contents":reranker[i]['rerank_passages'][0]+reranker[i]['rerank_passages'][1]+reranker[i]['rerank_passages'][2]
    }) 
with open("/root/shared-nvme/bm25-doc-bge/fl-xr-es-250301-reanker.json","w") as f:
    json.dump(a,f,ensure_ascii=False) 
print(1)  
#modelscope download --model BAAI/bge-m3 --local_dir /root/shared-nvme/bge-m3