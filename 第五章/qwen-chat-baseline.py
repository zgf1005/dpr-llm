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
    trust_remote_code=True,
   
).eval()
tokenizer = AutoTokenizer.from_pretrained("/root/shared-nvme/Qwen1.5-14B-Chat")
test_data = []
with open("/root/shared-nvme/es+dpr-0915.json","r") as f:
    test_data=json.load(f) 
a=[]
q=[]
for i in  range(len(test_data)):
    q.append(test_data[i]['ques_title'])
    
#valid_data = [json.loads(line) for line in open(f"/home/pod/shared-nvme/loc14_val.json",'r')]
for i in range(len(test_data)):
    r=test_data[i]['ans_contents']
    qus=q[i]+"请回答该问题,下面给出该问题的检索内容：\n"
    for j in range(len(r)):
        qus+=r[j]
    messages = [
        {"role": "system", "content": "你好"},
        {"role": "user", "content":qus}
        #{"role": "user", "content": prompt+test_data[i]["ques_title"]+"[输出]："}
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
    response=response.replace(",","，")  
    response=response.replace(qus,"，")
    response=response.replace("\n","")
    response=response.replace(";","；")
    print(response)
    print(i)
    a.append(response
    )
df_output = pd.DataFrame(a,columns=["candidates"])
df_output.to_csv(r"/root/shared-nvme/output-qwen1.5-law/candidateqwen-ques-1q3a-(es-dpr)-20241230.csv", index=False)
            
