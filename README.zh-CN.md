# AIMA AMD395 Qwen3.6 35B Windows Engine

这是一个面向 `Qwen3.6-35B-A3B-BF16`、AMD Ryzen AI Max+ 395（`gfx1151`）
的 TileRT 风格原生 Windows 推理引擎公开发布仓库。引擎针对 batch size 1，
核心运行时使用 C，外层工具使用 Rust。

> **发布状态：**目标运行时已经通过私有资格验收矩阵，但源码与二进制包
> 尚未公开。当前首个公开版本只包含发布契约、社区治理和开源就绪审计，
> 不是可安装的引擎版本。

[English](README.md)

## 已通过的目标验收

最近一次保留结果在项目 AMD395 主机上以原生 Windows 进程执行，使用真实
模型，并由独立 BF16 正确性基准提供权威边界。q8192 产品行结果如下：

| 指标 | 实测结果 | 验收边界 |
|---|---:|---:|
| TTFT | 3,852.909 ms | <= 4,187.416 ms |
| Prefill 吞吐 | 2,126.186 tok/s | >= 1,506.407 tok/s |
| Decode 吞吐 | 30.551 tok/s | >= 28.168 tok/s |
| TPOT | 32.732 ms | <= 35.502 ms |
| 模型与引擎加载 | 19,940.245 ms | <= 30,000 ms |

完整资格矩阵包含 12 条通过记录，覆盖 q8192 峰值、8k 到 128k 冷上下文、
16k 到 256k 共享前缀、512 token 解码和 SSE 流式输出，以及由外部 BF16
权威基准锚定的首 token 正确性。

这些数据是资格验收事实，并不表示当前 staging 仓库可以复现。公开复现仍
需要发布源码、生成 kernel 清单、Windows 可再分发运行库清单、精简证据包
和干净环境构建说明。

## 公开发布门槛

- [x] 原生 Windows 真实模型产品验收完成。
- [x] 产品行已经附加外部首 token 正确性权威边界。
- [x] 建立 Windows public repo 和 Apache-2.0 治理文件。
- [ ] 清除发布历史中的私有部署路径和地址。
- [ ] 公开 OpenAI API、MMLU-Pro 和产品矩阵的精简证据。
- [ ] 所有必要 provider 均可从干净 checkout 重建。
- [ ] 完成 Windows 可再分发组件及第三方许可证清单。
- [ ] 发布带校验和的源码 tag 和便携二进制包。

因此，目前可以公开说明“正在准备发布”，但还不能宣称这是一个可复现、
可安装的完整开源引擎。详细边界见
[开源就绪审计](docs/OPEN_SOURCE_READINESS.md)。

## 规划中的公共目录结构

仓库沿用配套 Linux 引擎的发布组织方式，但实现保持 Windows 原生：

- `engine/`：公共运行时 API 和编排；
- `native/`：C/C++/HIP provider 和生成的 `gfx1151` kernel；
- `scripts/`：MSVC、Rust、HIP、打包和验证入口；
- `benchmarks/`：附带正确性边界的精简产品证据；
- `packaging/`：便携 Windows 压缩包与许可证装配；
- `tests/`：CPU 安全检查与 Windows ABI 回归测试；
- `third_party/licenses/`：再分发组件的原始许可证文本。

本项目有意限定到特定模型和硬件，不是通用图运行时。模型权重不会随仓库
发布，用户必须自行取得并遵守相应许可证。

项目原创内容采用 Apache License 2.0。另有许可证或署名的文件继续遵守其
原始条款，详见 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
