# Stock Factor Strategy Research Framework

English | [中文](#中文简介)

## English
This repository is a reusable quantitative research framework for stock factor research, report generation, and optional strategy backtesting. The framework is config-driven, GitHub-safe, and built for long-term personal/research reuse.

### Language-Specific Full Docs
- Full English README: [README_EN.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/README_EN.md)
- Full Chinese README: [README_ZH.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/README_ZH.md)

### Quick Start
```bash
python3 -m pip install -r requirements.txt

python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

### Read Outputs in 30 Seconds
Open run directory in this order:
1. `README_FIRST.md`
2. `overview/factor_insights.csv`
3. `overview/factor_scorecard.csv`
4. `assets/key/`

### Minimal Docs (Simplified)
- [docs/user_guide.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/docs/user_guide.md)
- [docs/architecture.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/docs/architecture.md)

---

## 中文简介
这是一个面向股票因子研究的可复用工程框架，支持配置驱动运行、研究报告生成和可选回测。项目目标是把“可跑”升级为“可长期维护、可审计、可复现”。

### 完整版说明
- 英文完整版：[README_EN.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/README_EN.md)
- 中文完整版：[README_ZH.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/README_ZH.md)

### 快速开始
```bash
python3 -m pip install -r requirements.txt

python apps/run_from_config.py \
  --config configs/cs_factor.yaml \
  --set data.path=data/raw \
  --set factor.names='[factor_a,factor_b]'
```

### 结果最短阅读路径
每次运行目录按顺序看：
1. `README_FIRST.md`
2. `overview/factor_insights.csv`
3. `overview/factor_scorecard.csv`
4. `assets/key/`

### 精简后的文档入口
- [docs/user_guide.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/docs/user_guide.md)
- [docs/architecture.md](/home/oknotok/Projects/stock-factor-strategy-research-framework/docs/architecture.md)

### 一键清理输出
```bash
python apps/cleanup_outputs.py --root outputs --purge-all
```
