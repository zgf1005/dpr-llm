from transformers import AutoTokenizer, AutoModel
import json
import os
import pandas as pd
def write_file(filename,data):
    with open(filename,"w",encoding='utf-8' ) as f:
        f.write("candidates"+"\n")
        for i in range(len(data)):
            data[i]=data[i].replace('\n','')
        for i in range(len(data)):
            f.write(data[i]+"\n")
 
test_data = []
with open("/home/zhanggaofei/es+dpr/zgf-test-rerank-categories-front-20240514.json","r") as f:
    test_data=json.load(f) 


q=[]
relation=[]
for i in  range(len(test_data)):
    q.append(test_data[i]['ques_title'])
    
    
a=[] 
    

tokenizer = AutoTokenizer.from_pretrained("/home/zhanggaofei/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/zhanggaofei/chatglm2-6b", trust_remote_code=True, device='cuda',)
model = model.eval()

k=0

for i in range(len(test_data)):
    r=test_data[i]['ans_contents']
    qus=q[i]+"请回答该问题,下面给出该问题的检索内容：\n"
    for j in range(len(r)):
        qus+=r[j]
    response,history = model.chat(tokenizer, qus, [],max_length=2048)#,top_p=0.8,temperature=0.4)#temperature,repetition_penalty=1.5 越大生成越随机 
    response=response.replace(",","，")  
    response=response.replace(qus,"，")
    response=response.replace("\n","")
    response=response.replace(";","；")
    print(response)
    print(i)
    a.append(response)
    df_output = pd.DataFrame(a,columns=["candidates"])
    df_output.to_csv(r"/home/zhanggaofei/output-chatglm/candidatechatglm-dpr-ques+top3-20240515-(rerank-categories-front)-1q3a.csv", index=False)
            
#if not os.path.exists(f"/home/zhanggaofei/output-chatglm"):
#    os.makedirs(f"/home/zhanggaofei/output-chatglm")
    
#write_file(f"/home/zhanggaofei/output-chatglm/candidatechatglm-dpr-ques+top3-20240508-temperature=0.4-(categories_rear)-1q3a.csv", a)

