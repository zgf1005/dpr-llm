from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def write_file(filename,data):
    with open(filename,"w",encoding='utf-8' ) as f:
        f.write("candidates"+"\n")
        for i in range(len(data)):
            data[i]=data[i].replace('\n','')
            data[i]=data[i].replace(',','，')
        for i in range(len(data)):
            f.write(data[i]+"\n")
device = "cuda" # the device to load the model onto
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
model = AutoModelForCausalLM.from_pretrained(
    "/root/shared-nvme/Qwen1.5-14B-Chat",
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True,
    
)
tokenizer = AutoTokenizer.from_pretrained("/root/shared-nvme/Qwen1.5-14B-Chat")
test_data = []
with open("/root/shared-nvme/es+dpr-0915.json","r") as f:
    test_data=json.load(f) 
a=[]
for i in range(len(test_data)):
    prompt = f""""你是一个出色的关键词抽取器，你将会得到一段包含关键词的文本，你的任务是准确识别并提取所提
            供文本中的每个关键词。你的回复必须是一个关键词列表，列表内每一个元素分别对应于你所提取的关键词，不要回复其他任何以上未提及的东西。
        [回复格式]：关键词1、关键词2、.....
        这是两个例子：
        [输入]：问题：别人用我的信息买车，别人自己交钱，出了事，我会有责任吗
        [输出]:个人信息使用、付款责任、事故责任、法律责任、交通事故责任、交强险、个人信息保护法
        [输入]:问题：有四个半月女宝，男的不干，还砸东西打人。想离婚男方不同意把结婚证撕掉，
        [输出]:家庭暴力、离婚意愿、男方不同意、结婚证被毁、威胁
        请参照以上给出的两个例子，给下面输入文本的内容生成同义词：
        [输入]：问题："""
    messages = [
        {"role": "system", "content": "你好"},
        {"role": "user", "content": prompt+test_data[i]["ques_title"]+"\n[输出]:"}
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
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response=response.replace('[输出]:','')
    response=response.replace('[','')
    response=response.replace(']','')
    print(response)
    print(i)
    a.append({
        "ques_title":test_data[i]["ques_title"],
        "key":response
        }
    ) 
    

with open("/root/shared-nvme/output_qwen_key/fl-q+key-250207.json","w") as f:
    json.dump(a,f,ensure_ascii=False)   