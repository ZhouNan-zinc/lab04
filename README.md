# 作业四：Vector RAG / GraphRAG / WikiRAG

`hw4_rag.py` 是本次实验的唯一源码入口。默认使用
`paraphrase-multilingual-MiniLM-L12-v2`；检测到 CUDA 时自动把嵌入计算放到 GPU。
 正式结果统一使用
`sentence-transformers` + CUDA，写入 `results/formal_minilm/`。

```bash
cd lab04
source .venv/bin/activate
MODEL="$PWD/model-cache/paraphrase-multilingual-MiniLM-L12-v2"
python hw4_rag.py --mode eval --k 5 --emb "$MODEL" --device cuda --outdir results/formal_minilm
python hw4_rag.py --mode eval --k 1 --emb "$MODEL" --device cuda --outdir results/formal_minilm
python hw4_rag.py --mode eval --k 2 --emb "$MODEL" --device cuda --outdir results/formal_minilm
python hw4_rag.py --mode eval --k 3 --emb "$MODEL" --device cuda --outdir results/formal_minilm
python hw4_rag.py --mode vector --q Q01 --k 5 --emb "$MODEL" --device cuda
python hw4_rag.py --mode graph --q Q01 --scope local --k 5 --emb "$MODEL" --device cuda
python hw4_rag.py --mode graph --q Q01 --scope global --k 5 --emb "$MODEL" --device cuda
python hw4_rag.py --mode wiki --q Q11 --k 5 --emb "$MODEL" --device cuda
python hw4_rag.py --mrr_check 3,1,2
```

正式结果写入 `results/`，包括 JSON/CSV、逐题检索明细、运行设备与图表。15 道
内置题和 5 道自拟题均在源码 `QUESTIONS` 中，evidence 显式标注。未配置 API。
