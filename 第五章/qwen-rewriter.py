from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
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
    "/home/pod/shared-nvme/Qwen1.5-14B-Chat",
    torch_dtype="auto",
    device_map="auto",
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained("/home/pod/shared-nvme/Qwen1.5-14B-Chat")
test_data = []
with open("/root/shared-nvme/output_qwen_key/fl-q+key-250207.json","r") as f:
    test_data=json.load(f) 
a=[]
for i in range(len(test_data)):
    prompt = f"""你的任务是根据问题的关键词进行同义词扩展。这是例子：
        [输入]：问题：律师你好！我想问一下房产赠与但是没过户，怎么做可以要回 ？ 关键字：[房产赠与、未过户、要回房产、赠与合同、房产过户、法律效力、房屋产权、法律途径] 
        [输出]：房产赠与、未过户、要回房产、赠与合同、房产过户、法律效力、房屋产权、法律途径、撤销赠与、房产归属、合同有效性、产权变更、法律咨询、赠与撤销、房产纠纷、法律程序、房屋赠与、产权转移、法律保障、赠与行为、房产恢复。
        [输入]:问题：平房，房本是集体土地使用证，过户困难，可以打合同协议过户吗，因为必须要求本庄名下没有房产的才能过户，想问协议卖房可以吗，合同生效吗 ？关键字：[平房、集体土地使用证、过户困难、合同协议过户、本庄名下没有房产、协议卖房、合同生效]
        [输出]:单层住宅、一层房屋、独栋平房、单层建筑、集体土地产权证、农村集体土地证、集体土地使用权证、产权转移困难、房产交易障碍、所有权转移复杂
        、协议转让、合同转让、协议式产权转移、本村无房产、本集体无住房、本村无不动产、合同售房、协议出售、合同式房产交易、协议有效、合同具有法律效力、协议合法有效
        请参照以上给出的两个例子，给下面输入文本的内容生成同义词：
        [输入]：问题："""
    messages = [
        {"role": "system", "content": "你好"},
        {"role": "user", "content": prompt+test_data[i]["ques_title"]+"关键字："+test_data[i]["key"]+"\n[输出]:"}
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
        max_new_tokens=1024,
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
        "key-extend":response
        }
    )
if not os.path.exists(f"/home/pod/shared-nvme/output_qwen_key/"):
    os.makedirs(f"/home/pod/shared-nvme/output_qwen_key/")
with open("/home/pod/shared-nvme/output_qwen_key/fl-q+key-250207-rewriter.json","w") as f:
    json.dump(a,f,ensure_ascii=False)  