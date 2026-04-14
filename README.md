# PV-Meteorology-LLM

面向光伏与气象时空预测场景的垂直大模型工程实践，覆盖如下流程：

1. 领域数据构建
2. LoRA 微调
3. LoRA 合并基座模型
4. FastAPI 推理服务
5. Java 后端接入

## 项目说明

本项目基于 DeepSeek-R1-Distill-Qwen-1.5B 和 LLaMA-Factory，针对新能源与气象场景进行领域适配，目标是提升以下问题的回答质量：

1. 光伏预测中的时空建模设计
2. 扩散模型条件注入策略
3. 物理先验与数据驱动特征的取舍

## 环境要求

推荐 Linux + NVIDIA GPU 环境。

当前实践版本建议：

1. Python 3.11
2. PyTorch >= 2.6（已知安全限制，低版本可能触发 torch.load 限制）

## 一、训练环境准备

```bash
conda create -n llama-factory python=3.11 -y
conda activate llama-factory

git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

pip install -e ".[torch,metrics]"

# 按 CUDA 12.4 轮子安装（不改服务器驱动）
python -m pip install --upgrade --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

可选验证：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 二、下载基座模型

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U huggingface_hub
huggingface-cli download --resume-download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

## 三、数据集格式与注册

自定义数据集采用 Alpaca 格式（无 input 字段），示例：

```json
[
    {
        "instruction": "问题...",
        "output": "回答..."
    }
]
```

在 data/dataset_info.json 中注册时，建议显式映射列，避免 KeyError: 'input'：

```json
"PV": {
    "file_name": "PV.json",
    "columns": {
        "prompt": "instruction",
        "response": "output"
    }
}
```

## 四、启动微调

```bash
llamafactory-cli webui
```

在 WebUI 中：

1. 选择基座模型路径
2. 选择数据集 PV
3. 配置 LoRA 参数并开始训练

训练后输出目录通常在：

```text
/vision/lc/saves/<base_model_name>/lora/train_YYYY-MM-DD-HH-MM-SS
```

## 五、导出并合并模型

在 WebUI 的 Export 页签中，将 LoRA 适配器合并到基座模型，导出到例如：

```text
/vision/lc/LLM/my project/deepseek-r1-1.5b-merged
```

## 六、FastAPI 推理服务

```bash
conda create -n fastapi python=3.11 -y
conda activate fastapi
pip install fastapi uvicorn transformers torch safetensors sentencepiece protobuf
```

示例接口：

```python
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model_path = "/path/to/deepseek-r1-1.5b-merged"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")


@app.get("/generate")
async def generate_text(prompt: str):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs["input_ids"], max_length=512)
        return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
```

启动：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 七、Java Spring Boot 接入示例

```java
@Service
public class ChatServiceImpl implements ChatService {
        @Autowired
        private RestTemplate restTemplate;

        @Override
        public String callAiForOneReply(String prompt) {
                String url = String.format("http://GPU_SERVER_IP:8000/generate?prompt=%s", prompt);
                GenerateResponse response = restTemplate.getForObject(url, GenerateResponse.class);
                return response != null ? response.getGenerated_text() : "系统繁忙";
        }
}
```

## 常见问题

1. 报错 KeyError: 'input'
原因：数据是 instruction/output 格式，但未在 dataset_info.json 中显式列映射。
解决：为该数据集添加 prompt/response 映射。

2. 报错要求 torch >= 2.6
原因：torch.load 安全限制。
解决：仅升级 torch 到 2.6+ 即可，不需要改服务器 CUDA 驱动。
