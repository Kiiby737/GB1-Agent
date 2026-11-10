## 目录结构

```text
.
├── 1.data/
├── 2.train/
├── 3.agent/
├── 4.result/
├── 5.test/
│   ├── 1.baseline/
│   ├── 2.model_eval/
│   ├── 3.failure_analysis/
│   └── 4.result_presentation/
├── 6.knowledge/
├── figure/
├── Readme
├── README.md
├── requirements.txt
└── .gitignore
```

## 

使用已有M0与冻结预测池运行Agent

安装所需依赖：

```bash
python -m pip install -r requirements.txt
```

在仓库根目录配置路径，保持结果写入现有的 `4.result` 结构：

```bash
export GB1_V44_ROOT="$(pwd)"
export GB1_DATA_ROOT="$GB1_V44_ROOT"
export GB1_KG_DIR="$GB1_V44_ROOT/6.knowledge"
export GB1_AGENT_RESULT_DIR="$GB1_V44_ROOT/4.result"

export DEEPSEEK_API_KEY="填写你自己的API密钥"
export DEEPSEEK_BASE_URL="填写你实际使用的API服务base_url"
```



以相同IDGV起点分别运行三轮：

```bash
python 3.agent/run_agent_batch.py --start-round 1 --rounds 3 --knowledge off --start-sequence IDGV
python 3.agent/run_agent_batch.py --start-round 1 --rounds 3 --knowledge on --start-sequence IDGV
```

首次运行会重新建立起点记录。运行会产生新的结果，已有轮次不能直接覆盖。

