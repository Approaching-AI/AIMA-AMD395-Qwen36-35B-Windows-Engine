# AIMA AMD395 Qwen3.6 35B Windows Engine

这是一个面向 `Qwen3.6-35B-A3B-BF16` 和 AMD Ryzen AI Max+ 395
（`gfx1151`）的原生 Windows 推理引擎。它以 batch size 1 为核心，提供常驻
服务、OpenAI 兼容 HTTP API，以及结构化的 `qrt` 生命周期 CLI。

[English](README.md) ·
[配套 Linux 版本](https://github.com/skyguan92/AIMA-AMD395-Qwen36-35B-Linux-Engine)

## 已包含的能力

- C/C++/HIP 原生推理、模型加载和 `gfx1151` AOT kernel；
- `qrt serve/start/status/stop`；
- `/v1/models`、`/v1/completions`、`/v1/chat/completions`；
- 普通 JSON、SSE streaming、流式 function tool calling；
- tool result continuation 和最多 512 token 的确定性 greedy decode；
- 在总上下文上限内连续、任意的输入 token 长度；
- prefix cache seed、copy-on-write hit、resident hit 和污染隔离；
- 有界 FIFO 队列：一个请求执行，其他请求按顺序等待，队列满或超时返回
  明确的 `429`/`503`，不会无上限占用资源；
- 完整源码构建、验收、性能和 MMLU-Pro 数据。

本项目有意限定到单一模型族和硬件架构，不是通用图运行时。模型权重和 AMD
运行库不随仓库发布。

## 运行环境

- Windows 11 x64
- AMD Ryzen AI Max+ 395（`gfx1151`）
- AMD ROCm HIP SDK 7.1 与兼容驱动
- 用户自行取得的 `Qwen3.6-35B-A3B` BF16 模型目录

## 快速启动

从 [Releases](https://github.com/Approaching-AI/AIMA-AMD395-Qwen36-35B-Windows-Engine/releases)
下载 v1.0.0，或按 [源码构建说明](docs/BUILD.md)生成 runtime。完整启动命令见
英文 README；生命周期最常用的三条命令是：

```powershell
& .\engine\qrt.exe start <模型、provider、runtime.env 与端口参数>
& .\engine\qrt.exe status --state-file .\service.json
& .\engine\qrt.exe stop --state-file .\service.json --wait-seconds 120
```

`start` 完成模型预加载后会脱离当前终端并常驻；`status` 核验状态文件中的
精确 PID 和 health endpoint；`stop` 先关闭 admission，再等待活跃请求完成，
释放资源后退出。非 loopback 监听默认必须设置 `--api-key`。

OpenAI Python 客户端使用：

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="placeholder")
for chunk in client.chat.completions.create(
    model="qwen3.6-35b-a3b",
    messages=[{"role": "user", "content": "简要解释 prefix cache。"}],
    max_completion_tokens=128,
    temperature=0,
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

API、tool calling、错误码、队列和上下文限制详见 [API.md](docs/API.md)。

## 已公布的真实模型数据

| 指标 | 实测 | 验收边界 |
|---|---:|---:|
| 模型与引擎加载 | 19,940.245 ms | <= 30,000 ms |
| q8192 TTFT | 3,852.909 ms | <= 4,187.416 ms |
| q8192 prefill | 2,126.186 tok/s | >= 1,506.407 tok/s |
| decode | 30.551 tok/s | >= 28.168 tok/s |
| TPOT | 32.732 ms | <= 35.502 ms |

仓库公开 12 条性能记录、GB10 锚定的 token 正确性、长上下文 continuation、
OpenAI 功能验收，以及完整脱敏后的 MMLU-Pro candidate/reference 逐题数据。
MMLU-Pro 两端均为 `7,486 / 12,032`（`62.2174%`），12,032 题 projection
mismatch 为 0。详见 [性能](docs/PERFORMANCE.md)、[评测](docs/EVALUATION.md)
和 [benchmarks](benchmarks/README.md)。

## 构建与许可

Windows 全量构建：

```powershell
.\scripts\build-runtime.ps1 `
  -CkRoot C:\src\aiter-v0.1.13\3rdparty\composable_kernel `
  -OutDir build\runtime
```

跨平台 CPU-safe 检查运行 `make check`。目标构建还需要 VS Build Tools、
Rust、ROCm 7.1、WSL2 与 Triton 3.6，详见 [BUILD.md](docs/BUILD.md)。

项目原创内容采用 Apache-2.0；上游改编或再分发内容保持其原许可证。详见
[LICENSE](LICENSE)、[NOTICE](NOTICE) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
