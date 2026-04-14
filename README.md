# 🌞 PV-Meteorology-LLM：光伏与气象时空预测全栈大模型系统

<p align="center">
  <img src="https://img.shields.io/badge/Base_Model-DeepSeek_R1_1.5B-blue.svg" alt="Base Model">
  <img src="https://img.shields.io/badge/Fine_Tuning-LoRA-orange.svg" alt="Fine Tuning">
  <img src="https://img.shields.io/badge/Framework-LLaMA_Factory-green.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Serving-FastAPI_%7C_SpringBoot-red.svg" alt="Serving">
</p>

## 🎯 项目定位
本项目是一个针对**新能源与气象时空预测交叉领域**的垂直大模型全栈解决方案。
通用大模型在面对复杂的物理方程、扩散模型（Diffusion）架构设计时极易产生幻觉。本项目通过注入领域专家高质量数据，纠正了传统模型盲目使用全局太阳高度角/方位角（SZA/AZI）的误区，并打通了从 **“云端 GPU 微调 -> 模型合并 -> FastAPI 接口暴露 -> Java 后端业务集成 -> Web 前端展示”** 的完整工业级 AI 部署链路。

## ✨ 核心功能点
* **⚡️ 领域知识精准强化 (Domain-Specific LoRA)**：基于深层次的时空上下文融合策略微调，对 U-Net 主输入设计、特征淹没、归纳偏置等硬核物理与深度学习概念实现“零幻觉”精准问答。
* **🛠️ 低代码高效微调引擎**：深度集成 `LLaMA-Factory`，采用低秩矩阵分解（LoRA）进行部分参数微调，极大地降低了算力成本并防止过拟合。
* **🚀 高性能推理服务 (Model Serving)**：基于 `FastAPI` 构建轻量级 Python 推理网关，支持 GPU 显存动态分配与多轮对话生成。
* **🔗 业务解耦的全栈架构**：标准化的 RESTful API 设计，支持 `Spring Boot` 等企业级后端无缝接入，构建从底层算力到顶层 UI 的全业务闭环。

## 🏗️ 技术架构图

```mermaid
graph TD
    %% 前端与后端
    Client[Web前端 UI / 交互界面] -->|HTTP / JSON| Backend[Java Spring Boot 后端]
    Backend -->|业务处理 & 鉴权| Backend
    Backend -->|REST GET/POST 调用| API[FastAPI 推理网关]
    
    %% 模型层
    subgraph AI Server [云端 GPU 服务器 (AutoDL / 本地服务器)]
        API --> Engine[Transformers 推理引擎]
        Engine --> Model[已合并的大模型]
        
        subgraph Model Weights
            Base[基座模型: DeepSeek-R1-Distill-Qwen-1.5B]
            LoRA[LoRA 权重: PV-Meteorology 领域知识]
            Base -.融合.-> LoRA
        end
        Model --> Model Weights
    end
