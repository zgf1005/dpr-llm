import numpy as np
import logging
import spacy
import torch
from math import exp
from scipy.special import softmax
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
nlp = spacy.load("zh_core_web_sm")
model = AutoModelForCausalLM.from_pretrained(
    "/root/shared-nvme/Qwen1.5-14B-Chat",
    torch_dtype="auto",
    load_in_4bit=True,
    device_map="auto",
    use_flash_attention_2=True
).eval()
device = "cuda" # the device to load the model onto
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
tokenizer = AutoTokenizer.from_pretrained("/root/shared-nvme/Qwen1.5-14B-Chat")
def modifier(text, tokens, logprobs):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0
        for sid, sent in enumerate(sentences):
            pos = 0
            tr = tid
            while tr < len(tokens):
                apr = sent[pos:].find(tokens[tr])
                if apr == -1:
                    break
                pos = apr + len(tokens[tr])
                tr += 1
            probs = [1 - exp(v) for v in logprobs[tid:tr+1]]
            probs = np.array(probs)
            """ p = {
                "avg": np.mean,
                "max": np.max,
                "min": np.min,
            }.get(sentence_solver, lambda x: 0)(probs) """
            p=np.mean(probs)
            if p > 0.5: # hallucination
                # keep sentences before hallucination 
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                # replace all hallucinated tokens in current sentence with [xxx]
                curr = sentences[sid]
                pos = 0
                # # 这里改成了替换掉最大的那个，而不是所有的
                # max_prob = 0
                # for prob, tok in zip(probs, tokens[tid:tr+1]):
                #     max_prob = max(prob, max_prob)
                for prob, tok in zip(probs, tokens[tid:tr+1]):
                    apr = curr[pos:].find(tok) + pos
                    if prob > 0.5:
                    # if prob == max_prob:
                        curr = curr[:apr] + "[xxx]" + curr[apr+len(tok):]
                        pos = apr + len("[xxx]")
                    else:
                        pos = apr + len(tok)
                return prev, curr, True
            tid = tr + 1
        
        # No hallucination
        return text, None, False
    
    
def generate(input_text, max_length, return_logprobs=False):
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(device)
        input_length = input_ids.shape[1]
        attention_mask= torch.ones(input_ids.shape,dtype=torch.long,device=device)

        if return_logprobs:
            outputs = model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                output_scores = True,
            )
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )

            generated_tokens = outputs.sequences[:, input_length:]
            im_end_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')

            # 过滤掉 <|im_end|> 的 token ID
            filtered_tokens = [token for token in generated_tokens[0] if token != im_end_token_id]

            # 解码过滤后的 token
            text = tokenizer.decode(filtered_tokens)
            #text = tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)
            
            tokens = [tokenizer.decode(t) for t in filtered_tokens]
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs[:-1]]
            assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )
            generated_tokens = outputs[:, input_length:]
            text = tokenizer.decode(generated_tokens[0])
            return text, None, None


test_data = []
with open("/root/shared-nvme/zgf_test_20231027.json","r") as f:
    test_data=json.load(f)
re_w=[]
with open("/root/shared-nvme/output_qwen_key/question+key-rewriter.json","r") as f:
    re_w=json.load(f)
    
a=[]
q=[]
for i in  range(len(test_data)):
    q.append(test_data[i]['ques_title'])
for i in range(len(test_data)):
    r=test_data[i]['ans_contents']
    qus="给出问题和该问题相关的检索内容，请根据检索内容回答问题，问题："+q[i]+"检索内容："
    for j in range(len(r)):
        qus+=r[j]
    messages = [
        {"role": "system", "content": "你好"},
        {"role": "user", "content": q[i]}
    ] 
   
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = tokenizer.encode(text,return_tensors='pt')
    attention_mask= torch.ones(input_ids.shape,dtype=torch.long,device=device)
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=2048,
        pad_token_id=tokenizer.eos_token_id,
        
    )
    generated_id = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response = tokenizer.batch_decode(generated_id, skip_special_tokens=True)[0] 
    a.append({
        "ques_title":test_data[i]["ques_title"],
        "answer":response
    })
    print(i)
    """ new_text, tokens, logprobs = generate(
                text, 
                2048, 
                return_logprobs=True
            )
    new_text
    
    ptext, curr, hallucination = modifier(new_text, tokens, logprobs)  
    if(hallucination):
        print(1)
    a.append({
        "ques_title":test_data[i]["ques_title"],
        "hallucination":hallucination,
        "new_text":new_text,
        "ptext":ptext,
        "curr":curr
    })
    print(i) """
if not os.path.exists(f"/home/pod/shared-nvme/output_qwen_key/"):
    os.makedirs(f"/root/shared-nvme/output_qwen_key")
with open("/root/shared-nvme/output_qwen_key/hallucination-q-20241222.json","w") as f:
    json.dump(a,f,ensure_ascii=False)     
   
   