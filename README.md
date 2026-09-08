[English](./README.en.md) · [Website](https://zerorun.lei6393.com) · [GitHub](https://github.com/SuperMarioYL/zerorun)

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/hero-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/hero-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/hero-dark.svg">
  <img src="./assets/presentation/hero-light.svg" width="960" alt="Hero diagram">
</picture>

# zerorun

**运行模型前先检查启动计划。**

ZeroRun 将 GPU 元数据与版本化配置登记表比较，按声明启发式排序可行变体，并展示选中的启动参数。

## 为什么需要它

模型启动需要选择权重格式、后端和匹配参数。将选择保存在登记表中，便于检查决策，并可在下载或启动前 dry-run。

- **解释可行性** — 被拒绝变体包含需求差额。
- **版本化配置** — 模型与后端选择保存在 YAML。
- **启动前检查** — dry-run 展示计划而不启动推理。

## 架构

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/architecture-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/architecture-dark.svg">
  <img src="./assets/presentation/architecture-light.svg" width="960" alt="Architecture diagram">
</picture>

probe 获取 GPU 数据或使用显式模拟；登记表加载器校验条目，picker 检查配置显存和计算能力下限并排序，launcher 构造后端参数并可启动独立进程。

| 组件 | 职责 |
| --- | --- |
| `GPU metadata` | src/zerorun/probe.py |
| `Model registry` | registry/schema.py |
| `Heuristic picker` | src/zerorun/picker.py |
| `Launch arguments` | src/zerorun/launcher.py |

## 安装与快速上手

使用仓库清单指定的运行时版本构建，并在仓库根目录运行示例。

```bash
git clone https://github.com/SuperMarioYL/zerorun.git
cd zerorun
uv venv .venv
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate
```

对需要 16GB 与 32GB 的两个明确演示变体使用模拟 24GB GPU，检查 picker 结果。

```bash
.venv/bin/python examples/presentation-demo.py
```

## 实际运行示例

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/process-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/process-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/process-dark.svg">
  <img src="./assets/presentation/process-light.svg" width="960" alt="Process diagram">
</picture>

The 16GB-floor variant is selected and the 32GB-floor variant is reported infeasible under the synthetic input.

```text
{"quant": "Q4_K_M", "feasible": true, "reasons": ["score: accuracy 0.80 + backend 0.05 + vram-safety 0.020 = 0.870"]}
{"quant": "Q8_0", "feasible": false, "reasons": ["VRAM short: have 24.0 GB, need 32 GB (-8.0 GB)", "infeasible \u2014 listed for diagnostics only"]}
selected: Q4_K_M
```

完整命令与输出保存在 [docs/demo-results.json](./docs/demo-results.json). 输入和复现代码均随仓提供。

![已有终端录制](./assets/demo.gif)

保留已有录制供参考；上方文字示例给出当前可复现的操作。

## 用法

CLI 提供以下操作。示例之外的命令需要替换成你的文件路径或标识。

```bash
zerorun --list --gpu "24:8.9"
zerorun <registered-alias> --dry-run --gpu "24:8.9"
# Use a validated registry entry and installed backend for real execution:
zerorun <registered-alias>
```

## 配置

ZERORUN_REGISTRY_DIR 或 registry-dir 选项指定自定义登记表。条目定义 model、hf_id、quant/backend、min_vram_gb、compute_cap_min、args 与 chat_template_ref。--backend 和 --quant 约束选择，--dry-run 阻止启动。

## 集成与职责分工

<picture>
  <source media="(max-width: 600px) and (prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-mobile-dark.svg">
  <source media="(max-width: 600px)" srcset="./assets/presentation/integrations-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="./assets/presentation/integrations-dark.svg">
  <img src="./assets/presentation/integrations-light.svg" width="960" alt="Integrations diagram">
</picture>

以下路径已有源码实现。按任务选择输入，并把生成的结果与项目一起保存。

| 路径 | 已实现职责 |
| --- | --- |
| GPU probe / simulation | Hardware metadata input |
| Registry YAML | Variants and requirements |
| vLLM / llama.cpp | Supported launch backends |
| Dry-run | Inspect selection before launch |

## 限制与后续方向

- 登记表可行性基于声明下限，不是实测分配或运行验收；上下文、缓存和后端版本会改变需求。
- 排序系数是启发式，不是实测精度或吞吐；真实启动前应核实模型 ID、产物和聊天模板。
- 示例使用合成模型条目和模拟 GPU，不下载权重或启动后端。

登记表维护与在线后端验收需要持续进行；新增条目应有具体产物与目标环境证据。

## 许可与贡献

许可见 [LICENSE](./LICENSE). 反馈问题时请提供最小输入、执行命令和实际输出。
