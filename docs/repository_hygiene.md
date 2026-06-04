# 仓库提交边界

这个项目后续要作为面试展示仓库，提交边界需要非常清楚：提交源码、配置模板、测试、文档和少量精选 demo；不提交密钥、虚拟环境、运行产物、数据库和大模型文件。

## 应该提交

| 类型 | 路径 | 原因 |
|---|---|---|
| 核心源码 | `src/` | Agent、Orchestrator、tool loop、model provider、memory、evidence、wiki 等核心实现 |
| 配置样例 | `configs/` | 展示通用 DeepResearch 与 GIS/RS profile 的配置方式 |
| 运行脚本 | `scripts/` | 命令行 demo、trace 渲染、memory two-run demo |
| 单元测试 | `tests/unit/` | 证明关键模块可验证，不只是 demo 能跑 |
| 项目文档 | `README.md`, `docs/` | 面试官首先看的内容，应展示架构、能力、运行方式和结果 |
| 依赖声明 | `pyproject.toml`, `requirements*.txt` | 便于复现环境 |
| 环境变量模板 | `.env.template`, `.env.tools.template` | 告诉使用者需要哪些 key，但不泄露真实 key |
| 精选展示素材 | `docs/assets/`, `docs/demo/` | README 中展示图、报告片段、trace 示例 |

## 不应该提交

| 类型 | 路径 | 原因 |
|---|---|---|
| API Key | `.env`, `.env.local`, `.env.*.local` | 包含真实密钥，必须排除 |
| 虚拟环境 | `.venv/` | 体积大、平台相关、不可复用 |
| 运行输出 | `outputs/` | 每次 demo 都会产生大量报告、trace、JSONL，容易污染仓库 |
| 本地记忆库 | `data/`, `*.db`, `*.sqlite*` | 可能包含用户问题、报告内容和历史运行数据 |
| 安装日志 | `.setup-logs/`, `logs/` | 对展示无价值，且会产生噪声 |
| 模型权重 | `checkpoints/`, `models/`, `*.pt`, `*.bin`, `*.safetensors` | 体积大，且不适合放普通 Git 仓库 |
| 测试覆盖率输出 | `htmlcov/`, `.coverage*` | 自动生成，可重新生成 |
| 个人开发/学习记录 | `项目日志.md`, `分阶段改进执行计划.md`, `新项目功能总结与迁移建议.md`, `learning/`, `learning_note/` | 这些是个人开发过程资料，不适合作为面试展示仓库的一部分 |

## 新仓库建议提交策略

第一次提交只提交可展示的稳定内容：

```powershell
git add .gitignore README.md pyproject.toml requirements.txt requirements-minimal.txt `
  .env.template .env.tools.template configs scripts src tests docs

git commit -m "Initial clean GeoResearch Agent portfolio version"
```

提交前建议检查：

```powershell
git status --short --ignored
```

确认 `项目日志.md`、`learning/`、`outputs/`、`data/`、`.env`、`.venv/` 都处于 ignored 状态。
