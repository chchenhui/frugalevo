运行配置归档

目录：<实验批次>/<任务>/<算法>[__消融名称][__variantN].yaml
来源：runs/ 下各批次的 configs/**/*.yaml 和运行目录中的 config.yaml。
不收录搜索生成的 labels.yaml 或源码快照中的 models.yaml。

去重时递归忽略 seed/random_seed 字段，其余配置必须完全一致。
同组多 seed 仅保留最小 seed 的原始文件，保留其 seed 值；相同 seed
优先使用运行目录中的 config.yaml。重新运行其他 seed 时需自行修改。
若同组在预算、提示词或其他参数上有差异，分别保留为 __variantN。
批次名称沿用原目录；实际预算、模型等参数以 YAML 内容为准。

run_config_manifest.json 记录全部来源、合并的 seeds 和代表文件。
这些 YAML 是原始配置的逐字复制，未改动 runs 中的文件。
