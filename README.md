# 🚀 Awesome Local AI & Model Context Protocol (MCP) Architectures (2026)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Gumroad Store](https://img.shields.io/badge/Gumroad-Peterverse-blue.svg)](https://peterverse734.gumroad.com/)
[![Payhip Store](https://img.shields.io/badge/Payhip-Store-orange.svg)](https://payhip.com/peterverse21)

> **Curated production architectures, VRAM sizing calculus, Docker compose templates, and Model Context Protocol (MCP) implementations for modern open-weight LLMs (DeepSeek R1, Llama 3.3, vLLM, Ollama, FastMCP).**

---

## 📚 Official Engineering Blueprints & Master Manuals
Developed by **John Peter** and published by **Peterverse**:

| Blueprint | Focus Area | Pages | Download Links |
|---|---|:---:|:---:|
| **The Local AI & Open-Weight Blueprint** | vLLM Serving, AWQ/GGUF, VRAM Calculus, Zero-Cost Cloud | 21 Pages | [Gumroad ($39)](https://peterverse734.gumroad.com/l/local-ai-production-blueprint) \| [Payhip](https://payhip.com/peterverse21) |
| **The MCP & AI Coding Agent Blueprint** | Custom MCP Servers, JSON-RPC, Tool Sandboxing, AST | 19 Pages | [Gumroad ($39)](https://peterverse734.gumroad.com/l/mcp-ai-coding-agent-blueprint) \| [Payhip](https://payhip.com/peterverse21) |
| **The Autonomous AI Agents Blueprint** | ReAct Loops, Swarm Topologies, Memory Sandboxing | 66 Pages | [Gumroad ($69)](https://peterverse734.gumroad.com/l/gdjno) \| [Payhip](https://payhip.com/peterverse21) |
| **The Complete Developer Master Suite** | **All 8 Production Manuals + Complete Codebase** | Master Suite | [Gumroad ($79 Bundle)](https://peterverse734.gumroad.com/l/mdpvm) \| [Payhip](https://payhip.com/peterverse21) |

---

## ⚡ Quick Reference: GPU VRAM Calculus Formula

To prevent **CUDA Out of Memory (OOM)** exceptions during high-concurrency LLM inference:

$$\text{VRAM}_{\text{Total}} = M_{\text{weights}} + M_{\text{kv\_cache}} + M_{\text{activations}} + M_{\text{cuda\_overhead}}$$

### Where:
1. **$M_{\text{weights}}$:** $(\text{Parameters} \times \text{Bytes per Param}) \times 1.05$
   - 4-bit (AWQ/GGUF): $\sim 0.55\text{ GB per billion parameters}$
   - 8-bit (FP8/INT8): $\sim 1.05\text{ GB per billion parameters}$
   - 16-bit (FP16/BF16): $\sim 2.10\text{ GB per billion parameters}$
2. **$M_{\text{kv\_cache}}$:** $2 \times \text{Layers} \times \text{Heads} \times \text{HeadDim} \times \text{ContextTokens} \times \text{BatchSize} \times \text{Bytes}$
3. **$M_{\text{cuda\_overhead}}$:** $0.6\text{ GB to } 1.2\text{ GB}$ (PyTorch / CUDA runtime context)

### VRAM Hardware Allocation Matrix:
| Model Size | Quantization | Minimum VRAM | Recommended GPU | Inference Speed |
|---|---|---|---|---|
| **8B Params** | Q4_K_M (4-bit) | 6.5 GB | RTX 3060 / 4060 (12GB) | 50 - 75 t/s |
| **14B Params** | AWQ (4-bit) | 11.5 GB | RTX 4070 Ti / 4080 (16GB) | 45 - 65 t/s |
| **32B Params** | Q4_K_M (4-bit) | 22.0 GB | Single RTX 3090 / 4090 (24GB) | 25 - 35 t/s |
| **70B Params** | Q4_K_M (4-bit) | 44.0 GB | Dual RTX 3090 / 4090 (48GB Total) | 18 - 25 t/s |
| **70B Params** | FP8 / AWQ | 46.0 GB | Dual RTX 6000 Ada / A100 (80GB) | 40 - 60 t/s |

---

## 🐳 Production vLLM Docker Compose Configuration

Deploy high-throughput inference with **PagedAttention** and multi-GPU tensor parallelism:

```yaml
version: "3.8"

services:
  vllm-inference:
    image: vllm/vllm-openai:latest
    container_name: local-vllm-node
    runtime: nvidia
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
      - NCCL_DEBUG=INFO
    volumes:
      - /mnt/storage/models:/root/.cache/huggingface
    ipc: host
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      --model casperhansen/deepseek-r1-distill-qwen-14b-awq
      --tensor-parallel-size 2
      --gpu-memory-utilization 0.92
      --max-model-len 8192
      --port 8000
      --trust-remote-code
```

---

## 🔌 Minimal FastMCP Python Server Template

Build standardized Model Context Protocol tools in under 20 lines of Python:

```python
from mcp.server.fastmcp import FastMCP
import os

mcp = FastMCP("corporate-tools-server")

@mcp.tool()
async def read_project_file(file_path: str) -> str:
    """Reads a file safely within workspace boundaries."""
    safe_root = os.path.abspath(os.getcwd())
    resolved = os.path.abspath(os.path.join(safe_root, file_path))
    
    if not resolved.startswith(safe_root):
        return "Security Violation: Path traversal outside project root is prohibited."
    
    with open(resolved, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

---

## 🐘 PostgreSQL + pgvector Hybrid Search Schema

Air-gapped, zero-leakage local vector search using Hierarchical Navigable Small World (HNSW) indexing:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding vector(1024), -- Dimension for BAAI/bge-m3
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Fast sub-5ms cosine retrieval
CREATE INDEX ON document_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## 📖 Complete Engineering Handbooks
Need the full step-by-step implementations, test scripts, and enterprise security checklists?
* **Local AI & Open-Weight Production Manual:** [Download on Gumroad](https://peterverse734.gumroad.com/l/local-ai-production-blueprint)
* **Model Context Protocol & Coding Agent Blueprint:** [Download on Gumroad](https://peterverse734.gumroad.com/l/mcp-ai-coding-agent-blueprint)
* **Autonomous AI Agents Architecture (66 Pages):** [Download on Gumroad](https://peterverse734.gumroad.com/l/gdjno)
* **Payhip Digital Storefront:** [Visit Payhip](https://payhip.com/peterverse21)

---

## 📄 License
MIT © Peterverse & John Peter.
