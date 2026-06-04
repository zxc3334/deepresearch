# Demo 报告片段

> 这是用于 README 展示的摘要片段，不是完整运行产物。完整报告会生成到 `outputs/<run-id>/report_*.md`，该目录默认不提交。

## 查询

如何研究 2018-2024 年武汉城市扩张对地表热环境的影响？请给出数据选择、方法流程、验证方案和潜在风险。

## 输出摘要

系统将问题拆成若干个可并发执行的子任务：数据源检索、遥感方法检索、官方文档核验、风险检查和最终综合。每个子任务通过 tool-calling loop 调用外部搜索、论文检索、官方来源检索和网页读取工具，并把结果写入 evidence store 与 trace。

## 示例证据分级

| 证据等级 | 含义 | 示例 |
|---|---|---|
| `evidence_backed` | 有论文、官方文档或可追踪网页支持 | Landsat Collection 2 Level-2 LST 可作为地表温度产品来源 |
| `speculative` | 逻辑上可能成立，但当前证据不足 | 某个局部城市扩张结论需要本地样本和时序数据验证 |
| `rejected` | 与工具证据或领域约束冲突 | 直接用 Sentinel-2 反演 LST，因为 Sentinel-2 没有热红外波段 |

## 示例工具统计

| 指标 | 示例值 |
|---|---:|
| Trace events | 95 |
| Tool calls | 12 |
| Evidence-backed items | 12 |
| Speculative items | 4 |
| Rejected items | 1 |
