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
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("/home/pod/shared-nvme/Qwen1.5-14B-Chat")
test_data = []
with open("/home/pod/shared-nvme/output_qwen_key/medical_test_key.json","r") as f:
    test_data=json.load(f) 
a=[]
for i in range(len(test_data)):
    prompt = f""" 请根据给定的文本找出能回答问题的关键文本片段，如
果能够回答问题就输出答案相关的文本片段，文本中没有能
够回答问题的片段请输出[该文本不包含答案]，不要自己生
成答案。这是例子：
[文本]腺癌 是 一 种 常 发生 在 腮腺 以及 腭部 小涎腺 部位 的 恶性肿瘤 ， 患者 常见 为 中老年 ， 腺癌 发展 较 快 ， 而且 病程 比较 短 ， 可以 侵害 皮肤 肌肉 ， 骨组织 等等 ， 在 治疗 方面 可以 选择 手术 切除 ， 如果 患者 不 适宜 做 手术 ， 也 可以 选择 放疗 化疗 缓解 病情, 这种 疾病 主要 是 皮肤 组织 的 一些 恶性肿瘤 而 造成 的 一般 我们 都会 称 作为 癌症 ， 癌症 的 早期 进行 控制 是 可以 延长 自己 的 寿命 ， 如果 已经 出现 了 扩散 的 情况 ， 就 只能 通过 化疗 或者 手术 的 方式 尽量 的 控制, 腺瘤 是 由 腺上皮 产生 的 良性肿瘤 ， 可以 在 内分泌 腺体 中 发现 ， 如 乳腺 、 垂体 、 甲状腺 、 卵巢 、 胃 、 肠 、 肝 等 ， 发育 缓慢 ， 形成 局部 结节 、 息肉样 或 乳头状 表面 ， 活检 是 可行 的 ， 根据 临床 表现 和 相关 检查 不难 获得 诊断
[待检查的问题]肿瘤 是 癌症 吗 ？
[输出]这种疾病主要是皮肤组织的一些恶性肿瘤而造成的，一般我们都会称作为癌症。
[文本]如果 只有 身上 瘙痒 ， 而 没有 发现 皮疹 的 情况 ， 属于 瘙痒症 ， 如果 身上 起 红斑 风团 “ 疙瘩 ” ， 一会 儿起 ， 一会儿 消则 属于 荨麻疹 ， 荨麻疹丸 有 抗过敏 的 作用 ， 可以 先 吃 几 天 ， 观察 疗效, 强阳性 有 可能 激素 还 没有 完全 撤退 ， 这样 的 情况 建议 同 房 尝试 一下 ， 建议 平时 清淡 饮食 ， 规律 生活 ， 少 吃 辛辣 刺激性 的 及 生冷 的 食物, 你 是 想咨询 一下 糖尿病 是 吧 ！ 甜蜜期 是 指 ， 控制 的 好的 那一段 时间 吗 ？ 医学 上 没有 这个 述语 ， 建议 ， 得了 糖尿病 以后 ， 需 注意 低 盐 低脂饮食 ， 低糖 饮食 ， 多 加 注意 锻炼 身体 ， 定期 监测 血糖 变化 ， 控制 并发症 ， 就是 管住 嘴 ， 迈 开腿 ， 麻烦 医生 评价 下 ” 这个 问题 的 建议。
[待检查的问题]手臂 麻木 ， 疼痛 ， 上 医院 检查 说 神经 受损 ，
[输出]该文本不包含答案
请根据以上两个例子生成响应：
[文本]：
"""
    messages = [
        {"role": "system", "content": "你好"},
        {"role": "user", "content": prompt+test_data[i]["ques_title"]+"[待检查的问题]："+test_data[i]["key"]+"\n[输出]:"}
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
with open("/home/pod/shared-nvme/output_qwen_key/question+key-rewriter-new.json","w") as f:
    json.dump(a,f,ensure_ascii=False)  