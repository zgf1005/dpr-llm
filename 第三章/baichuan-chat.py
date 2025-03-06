import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from transformers.generation.utils import GenerationConfig
CUDA_DEVICE = "cuda:0"
def write_file(filename,data):
    with open(filename,"w",encoding='utf-8' ) as f:
        f.write("candidates"+"\n")
        for i in range(len(data)):
            data[i]=data[i].replace('\n','')
            data[i]=data[i].replace(',','，')
        for i in range(len(data)):
            f.write(data[i]+"\n")
tokenizer = AutoTokenizer.from_pretrained("./baichuan-chat-7b", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("./baichuan-chat-7b",
                                             trust_remote_code=True,
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16
                                             ).half()

model.generation_config = GenerationConfig.from_pretrained("./baichuan-chat-7b")#baichuan-inc/Baichuan2-7B-Chat
model = model.eval()
#model.generation_config.repetition_penalty=1.3
model.generation_config.max_length=2048
model.generation_config.temperature=0.6
test_data = []
with open("/home/zhanggaofei/es+dpr/zgf_test_es_categories_rear_20231028.json","r") as f:
    test_data=json.load(f) 

q=[]
relation=[]
for i in  range(len(test_data)):
    q.append(test_data[i]['ques_title'])
    
a=[] 
k=0 
for i in range(len(test_data)):
    r=test_data[i]['ans_contents']
    qus=q[i]+"请回答该问题,下面给出该问题的检索内容：\n"
    for j in range(len(r)):
        qus+=r[j]
    messages=[]
    messages.append({"role": "user", "content": qus})#,"parameters": {"temperature": 0.8} #,"top_k": 10
 
    response = model.chat(tokenizer, messages)
    response=response.replace(",","，")
    response=response.replace("\n","")
    print(response)
    print(i)
    a.append(response)
    
if not os.path.exists(f"/home/heb/zgf/dpr+es/output_baichuan/"):
    os.makedirs(f"/home/heb/zgf/dpr+es/output_baichuan/")
write_file(f"/home/heb/zgf/dpr+es/output_baichuan/candidatebaichuan-dpr-ques+top3-20240226-temperature=0.6-(es_categories_rear)-1q3a.csv", a)
    
torch.cuda.empty_cache()


# RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cuda:1!


#chatbot[-1] = (parse_text(input), parse_text(response)) 

# 释放GPU内存：在每次模型计算后，您可以使用torch.cuda.empty_cache()方法释放GPU上的内存，以便为后续计算腾出空间。




