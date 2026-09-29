from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results"

CORPUS = {
    "doc01": "云山大学位于岭南省云州市，建于 1958 年，是省属重点大学，现有全日制本科生约 2.1 万人。校训为「格物致知」。",
    "doc02": "人工智能学院成立于 2018 年，首任及现任院长为王启明教授。学院下设机器学习系、智能科学系、认知计算系三个系。",
    "doc03": "计算机学院前身为 1985 年成立的计算机系，现任院长陈国峰教授，设计算机科学与技术、软件工程两个本科专业。",
    "doc04": "人工智能学院教师：李文瀚，教授，2019 年入职，研究方向为检索增强生成，主讲《大模型通识课》；赵婉晴，副教授，研究方向为知识图谱，主讲《知识图谱导论》；孙浩然，讲师，研究方向为强化学习。",
    "doc05": "计算机学院教师：周天宇，教授，研究方向为分布式系统；林小雨，副教授，研究方向为数据库系统。",
    "doc06": "《大模型通识课》课程代码 AI2101，3 学分、32 学时，春季学期开设，授课教师李文瀚，面向全校本科生，无先修课程要求。",
    "doc07": "《知识图谱导论》课程代码 AI3305，2 学分，秋季学期开设，授课教师赵婉晴，面向人工智能学院研究生。",
    "doc08": "选课规则：本科生每学期最多修 30 学分；GPA 低于 2.0 给予学术警告。先修要求：《知识图谱导论》需先修《数据结构》，《大模型通识课》无先修要求。",
    "doc09": "奖学金体系：国家奖学金 8000 元/年，评定比例约 2%；校长奖学金 20000 元/年，全校每年 10 人；云山一等奖学金 3000 元/年，比例约 5%。",
    "doc10": "科研平台：认知计算全国重点实验室依托人工智能学院建设；岭南超算中心由云山大学与省科技厅共建，计算机学院参与运行管理。",
    "doc11": "学生社团：AI 协会，指导教师李文瀚，每周三晚组织论文研读；机器人战队，指导教师孙浩然，每年参加全国机器人大赛。",
    "doc12": "校历：春季学期 3 月 2 日开学、7 月 5 日放暑假；秋季学期 9 月 1 日开学、次年 1 月 15 日放寒假。",
    "doc13": "图书馆藏书 380 万册，开放时间为每天 7:00-22:00；人工智能分馆位于理科楼 B 座 3 层，收藏大模型与智能体专题图书。",
    "doc14": "校园交通：地铁 3 号线「云大站」距东门 200 米；校内校巴共 5 条线路，10 分钟一班。",
    "doc15": "国际交流：学校与 12 个国家的 47 所高校签有交换协议；人工智能学院与新加坡南洋理工大学有本科联合培养项目。",
}

QUESTIONS = [
    {"id": "Q01", "type": "multi", "q": "《大模型通识课》授课教师所在学院的行政负责人是谁？", "evidence": ["doc04", "doc06", "doc02"]},
    {"id": "Q02", "type": "multi", "q": "AI 协会的指导教师主讲课程的课程代码是什么？", "evidence": ["doc11", "doc04", "doc06"]},
    {"id": "Q03", "type": "multi", "q": "认知计算全国重点实验室依托的学院成立于哪一年？", "evidence": ["doc10", "doc02"]},
    {"id": "Q04", "type": "multi", "q": "《知识图谱导论》授课教师的研究方向是什么？", "evidence": ["doc07", "doc04"]},
    {"id": "Q05", "type": "multi", "q": "机器人战队的指导教师的研究方向是什么？", "evidence": ["doc11", "doc04"]},
    {"id": "Q06", "type": "fact", "q": "云山大学的校训是什么？", "evidence": ["doc01"]},
    {"id": "Q07", "type": "fact", "q": "国家奖学金的金额是多少？", "evidence": ["doc09"]},
    {"id": "Q08", "type": "fact", "q": "图书馆每天几点到几点开放？", "evidence": ["doc13"]},
    {"id": "Q09", "type": "fact", "q": "人工智能学院成立于哪一年？", "evidence": ["doc02"]},
    {"id": "Q10", "type": "fact", "q": "春季学期什么时候开学？", "evidence": ["doc12"]},
    {"id": "Q11", "type": "global", "q": "云山大学有哪些科研平台？分别依托谁建设？", "evidence": ["doc10", "doc02", "doc03"]},
    {"id": "Q12", "type": "global", "q": "学校的学生奖励体系包含哪些项目？", "evidence": ["doc09"]},
    {"id": "Q13", "type": "global", "q": "人工智能学院有哪些教师？各自做什么研究方向？", "evidence": ["doc04"]},
    {"id": "Q14", "type": "global", "q": "本科生选课有哪些限制和先修要求？", "evidence": ["doc08"]},
    {"id": "Q15", "type": "global", "q": "对想深入学习大模型的学生，学校提供哪些课程和课外活动？", "evidence": ["doc06", "doc11"]},
    # 自拟题
    {"id": "Q16", "type": "fact", "q": "云大站距离学校东门多远？", "evidence": ["doc14"]},
    {"id": "Q17", "type": "fact", "q": "校长奖学金每年全校有多少名额？", "evidence": ["doc09"]},
    {"id": "Q18", "type": "multi", "q": "AI 协会指导教师所在学院的院长是谁？", "evidence": ["doc11", "doc04", "doc02"]},
    {"id": "Q19", "type": "global", "q": "学校有哪些学生学习与校园生活支持资源？", "evidence": ["doc09", "doc11", "doc13"]},
    {"id": "Q20", "type": "global", "q": "学校有哪些科研合作与国际交流安排？", "evidence": ["doc10", "doc15"]},
]

PRE_TRIPLES = [
    ("李文瀚", "任职于", "人工智能学院", "doc04"), ("李文瀚", "职称", "教授", "doc04"),
    ("李文瀚", "入职年份", "2019 年", "doc04"), ("李文瀚", "研究方向", "检索增强生成", "doc04"),
    ("李文瀚", "主讲", "大模型通识课", "doc06"), ("李文瀚", "指导", "AI 协会", "doc11"),
    ("赵婉晴", "任职于", "人工智能学院", "doc04"), ("赵婉晴", "研究方向", "知识图谱", "doc04"),
    ("赵婉晴", "主讲", "知识图谱导论", "doc07"), ("孙浩然", "任职于", "人工智能学院", "doc04"),
    ("孙浩然", "研究方向", "强化学习", "doc04"), ("孙浩然", "指导", "机器人战队", "doc11"),
    ("人工智能学院", "院长", "王启明", "doc02"), ("人工智能学院", "成立于", "2018 年", "doc02"),
    ("人工智能学院", "下设系", "机器学习系", "doc02"), ("人工智能学院", "下设系", "智能科学系", "doc02"),
    ("人工智能学院", "下设系", "认知计算系", "doc02"), ("认知计算全国重点实验室", "依托", "人工智能学院", "doc10"),
    ("岭南超算中心", "参与共建", "计算机学院", "doc10"), ("计算机学院", "院长", "陈国峰", "doc03"),
    ("计算机学院", "前身", "计算机系", "doc03"), ("周天宇", "任职于", "计算机学院", "doc05"),
    ("周天宇", "研究方向", "分布式系统", "doc05"), ("林小雨", "任职于", "计算机学院", "doc05"),
    ("林小雨", "研究方向", "数据库系统", "doc05"), ("大模型通识课", "课程代码", "AI2101", "doc06"),
    ("大模型通识课", "学分", "3 学分", "doc06"), ("大模型通识课", "开设学期", "春季学期", "doc06"),
    ("知识图谱导论", "课程代码", "AI3305", "doc07"), ("知识图谱导论", "先修课程", "数据结构", "doc08"),
    ("AI 协会", "指导教师", "李文瀚", "doc11"), ("机器人战队", "指导教师", "孙浩然", "doc11"),
    ("国家奖学金", "金额", "8000 元/年", "doc09"), ("校长奖学金", "金额", "20000 元/年", "doc09"),
    ("云山一等奖学金", "金额", "3000 元/年", "doc09"), ("云山大学", "校训", "格物致知", "doc01"),
    ("云山大学", "建校于", "1958 年", "doc01"), ("云山大学", "位于", "岭南省云州市", "doc01"),
    ("图书馆", "藏书量", "380 万册", "doc13"), ("春季学期", "开学日期", "3 月 2 日", "doc12"),
]

COMMUNITY_SUMMARIES = [
    ("社区 C1 教学与课程：人工智能学院设三个系，开设《大模型通识课》（AI2101，李文瀚主讲）与《知识图谱导论》（AI3305，赵婉晴主讲），AI3305 需先修《数据结构》。", ["doc02", "doc04", "doc06", "doc07", "doc08"]),
    ("社区 C2 科研与国际：认知计算全国重点实验室依托人工智能学院，岭南超算中心由计算机学院参与共建；学院与新加坡南洋理工大学有联合培养项目。", ["doc02", "doc03", "doc10", "doc15"]),
    ("社区 C3 学生生活：奖学金分国家（8000 元）、校长（20000 元）、云山一等（3000 元）三级；社团有 AI 协会与机器人战队，图书馆藏书 380 万册、7:00-22:00 开放。", ["doc09", "doc11", "doc13"]),
]

PRE_ENTRIES = [
    ("云山大学", "云山大学位于岭南省云州市，建于 1958 年，省属重点大学，本科生约 2.1 万人，校训「格物致知」。地铁 3 号线「云大站」距东门 200 米，校巴 5 条线路。校历：春季学期 3 月 2 日开学。", ["doc01", "doc12", "doc14"]),
    ("人工智能学院", "人工智能学院成立于 2018 年，院长王启明教授，下设机器学习系、智能科学系、认知计算系。认知计算全国重点实验室依托本院建设。与新加坡南洋理工大学有本科联合培养项目。", ["doc02", "doc10", "doc15"]),
    ("计算机学院", "计算机学院前身是 1985 年成立的计算机系，院长陈国峰教授，设计算机科学与技术、软件工程两个专业。参与岭南超算中心运行管理。", ["doc03", "doc10"]),
    ("人工智能学院教师", "李文瀚：教授，2019 年入职，研究检索增强生成，主讲《大模型通识课》，指导 AI 协会。赵婉晴：副教授，研究知识图谱，主讲《知识图谱导论》。孙浩然：讲师，研究强化学习，指导机器人战队。", ["doc04", "doc11"]),
    ("计算机学院教师", "周天宇：教授，研究分布式系统。林小雨：副教授，研究数据库系统。", ["doc05"]),
    ("大模型通识课", "《大模型通识课》（AI2101）：3 学分 32 学时，春季学期开设，李文瀚主讲，面向全校本科生，无先修要求。", ["doc06"]),
    ("知识图谱导论", "《知识图谱导论》（AI3305）：2 学分，秋季学期，赵婉晴主讲，面向人工智能学院研究生，需先修《数据结构》。", ["doc07", "doc08"]),
    ("奖学金体系", "国家奖学金 8000 元/年（约 2%）；校长奖学金 20000 元/年（全校 10 人）；云山一等奖学金 3000 元/年（约 5%）。", ["doc09"]),
    ("学生社团与校园生活", "AI 协会：指导教师李文瀚，每周三晚论文研读。机器人战队：指导教师孙浩然，参加全国机器人大赛。图书馆藏书 380 万册，每天 7:00-22:00 开放，人工智能分馆在理科楼 B 座。", ["doc11", "doc13"]),
]


def chunk_docs(size: int = 120, overlap: int = 20) -> list[tuple[str, str]]:
    if overlap >= size:
        raise ValueError("overlap 必须小于 size")
    out = []
    for doc_id, text in CORPUS.items():
        for start in range(0, len(text), size - overlap):
            out.append((doc_id, text[start:start + size]))
            if start + size >= len(text):
                break
    return out


class Embedder:
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.backend = "sentence-transformers"
        self.device = "cpu"
        self.model = None
        try:
            import torch
            requested = device
            if requested == "auto":
                requested = "cuda" if torch.cuda.is_available() else "cpu"
            if requested == "cuda" and not torch.cuda.is_available():
                requested = "cpu"
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=requested)
            self.backend = "sentence-transformers"
            self.device = requested
            print(f"[embed] 已加载 {model_name} / device={self.device}")
        except Exception as exc:
            raise RuntimeError(f"嵌入模型加载失败：{model_name}（{type(exc).__name__}: {exc}）。请先完成模型下载后再运行正式实验。") from exc

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        arr = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(arr, dtype=np.float32)


def _dedup(scored: Iterable[tuple[str, float]], k: int) -> list[str]:
    docs, seen = [], set()
    for doc, _score in scored:
        if doc not in seen:
            docs.append(doc)
            seen.add(doc)
            if len(docs) >= k:
                break
    return docs


class VectorRAG:
    name = "Vector RAG"

    def __init__(self, emb: Embedder):
        self.emb = emb
        self.chunks = chunk_docs()
        self.M = emb.encode([text for _, text in self.chunks])

    def scored(self, query: str) -> list[tuple[str, float, str]]:
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        order = np.argsort(-scores)
        return [(self.chunks[i][0], float(scores[i]), self.chunks[i][1]) for i in order]

    def retrieve(self, query: str, k: int = 5, scope: str | None = None) -> list[str]:
        return _dedup(((d, s) for d, s, _ in self.scored(query)), k)


class GraphRAG:
    name = "GraphRAG"

    def __init__(self, emb: Embedder):
        import networkx as nx
        self.emb = emb
        self.G = nx.DiGraph()
        for h, r, t, src in PRE_TRIPLES:
            self.G.add_edge(h, t, rel=r, src=src)
        try:
            self.communities = list(nx.community.greedy_modularity_communities(self.G.to_undirected()))
        except Exception:
            self.communities = [set(self.G.nodes)]
        self.summ_M = emb.encode([s for s, _ in COMMUNITY_SUMMARIES])

    def _entities(self, query: str, topn: int = 3) -> list[str]:
        return sorted([n for n in self.G.nodes if n in query], key=len, reverse=True)[:topn]

    def _local_scored(self, query: str) -> list[tuple[str, float, str]]:
        entities = self._entities(query)
        if not entities:
            return self._global_scored(query)
        seen, scored, frontier = set(), [], entities[:]
        for _ in range(3):
            nxt = []
            for entity in frontier:
                edges = list(self.G.out_edges(entity, data=True)) + list(self.G.in_edges(entity, data=True))
                for h, t, data in edges:
                    key = (h, data["rel"], t)
                    if key in seen:
                        continue
                    seen.add(key)
                    txt = f"{h} —{data['rel']}→ {t}"
                    scored.append((data["src"], 0.0, txt))
                    other = t if h == entity else h
                    if other not in frontier:
                        nxt.append(other)
            nxt.sort(key=lambda x: (self.G.out_degree(x) == 0, -self.G.degree(x)))
            frontier = nxt[:8]
        qv = self.emb.encode([query])[0]
        vals = self.emb.encode([txt for _, _, txt in scored]) @ qv
        best: dict[str, tuple[float, str]] = {}
        for (doc, _, txt), score in zip(scored, vals):
            if doc not in best or score > best[doc][0]:
                best[doc] = (float(score), txt)
        return [(doc, score, txt) for doc, (score, txt) in sorted(best.items(), key=lambda x: -x[1][0])]

    def _global_scored(self, query: str) -> list[tuple[str, float, str]]:
        qv = self.emb.encode([query])[0]
        sims = self.summ_M @ qv
        docs: list[tuple[str, float, str]] = []
        for idx in np.argsort(-sims)[:2]:
            summary, sources = COMMUNITY_SUMMARIES[int(idx)]
            for doc in sources:
                if not any(d == doc for d, _, _ in docs):
                    docs.append((doc, float(sims[idx]), summary))
        return docs

    def scored(self, query: str, scope: str = "local") -> list[tuple[str, float, str]]:
        return self._local_scored(query) if scope == "local" else self._global_scored(query)

    def retrieve(self, query: str, k: int = 5, scope: str = "local") -> list[str]:
        return _dedup(((d, s) for d, s, _ in self.scored(query, scope)), k)


class WikiRAG:
    name = "WikiRAG"

    def __init__(self, emb: Embedder):
        self.emb = emb
        self.M = emb.encode([body for _, body, _ in PRE_ENTRIES])

    def scored(self, query: str) -> list[tuple[str, float, str]]:
        qv = self.emb.encode([query])[0]
        scores = self.M @ qv
        out = []
        for idx in np.argsort(-scores)[:3]:
            title, body, sources = PRE_ENTRIES[int(idx)]
            for doc in sources:
                out.append((doc, float(scores[idx]), f"{title}：{body}"))

        best: dict[str, tuple[float, str]] = {}
        for doc, score, text in out:
            if doc not in best:
                best[doc] = (score, text)
        return [(d, s, t) for d, (s, t) in best.items()]

    def retrieve(self, query: str, k: int = 5, scope: str | None = None) -> list[str]:
        return _dedup(((d, s) for d, s, _ in self.scored(query)), k)


def evaluate(systems: dict[str, Any], k: int, questions: list[dict[str, Any]] = QUESTIONS, graph_scope: str = "local") -> dict[str, Any]:
    by_system: dict[str, Any] = {}
    for name, system in systems.items():
        rows = []
        for q in questions:
            docs = system.retrieve(q["q"], k, graph_scope) if name == "GraphRAG" else system.retrieve(q["q"], k)
            evidence = list(q["evidence"])
            hit = [d for d in evidence if d in docs[:k]]
            rank = next((i + 1 for i, d in enumerate(docs[:k]) if d in evidence), 0)
            rows.append({"id": q["id"], "type": q["type"], "question": q["q"], "evidence": evidence, "docs": docs, "hits": hit, "recall": len(hit) / len(evidence), "rank": rank, "rr": 1.0 / rank if rank else 0.0, "full": set(hit) == set(evidence)})
        groups = {}
        for tp in ("fact", "multi", "global", "all"):
            subset = rows if tp == "all" else [r for r in rows if r["type"] == tp]
            groups[tp] = {"n": len(subset), "recall": float(np.mean([r["recall"] for r in subset])), "mrr": float(np.mean([r["rr"] for r in subset])), "full": int(sum(r["full"] for r in subset))}
        by_system[name] = {"groups": groups, "rows": rows}
    return {"k": k, "graph_scope": graph_scope, "questions": [q["id"] for q in questions], "systems": by_system}


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def env_info(emb: Embedder) -> dict[str, Any]:
    info = {"python": sys.version, "platform": platform.platform(), "embed_backend": emb.backend, "embed_device": emb.device, "model": emb.model_name}
    try:
        import torch
        info.update({"torch": torch.__version__, "cuda_available": bool(torch.cuda.is_available()), "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception:
        info.update({"torch": None, "cuda_available": False, "cuda_device": None})
    return info


def build_systems(emb: Embedder, mode: str) -> dict[str, Any]:
    systems = {}
    if mode in ("eval", "vector"):
        systems["Vector RAG"] = VectorRAG(emb)
    if mode in ("eval", "graph"):
        systems["GraphRAG"] = GraphRAG(emb)
    if mode in ("eval", "wiki"):
        systems["WikiRAG"] = WikiRAG(emb)
    return systems


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(outdir: Path, result: dict[str, Any], title: str = "Recall@K") -> None:
    try:
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        systems = list(result["systems"])
        types = ["fact", "multi", "global"]
        x = np.arange(len(types)); width = 0.8 / max(1, len(systems))
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for i, name in enumerate(systems):
            vals = [result["systems"][name]["groups"][tp]["recall"] for tp in types]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=name)
        ax.set_xticks(x + width * (len(systems) - 1) / 2 - 0.4 + width / 2, ["fact", "multi", "global"])
        ax.set_ylim(0, 1.05); ax.set_ylabel("Recall"); ax.set_title(title); ax.legend(); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(outdir / "figure_recall.png", dpi=180); plt.close(fig)
    except Exception as exc:
        (outdir / "plot_error.txt").write_text(str(exc), encoding="utf-8")


def run_eval(args: argparse.Namespace, emb: Embedder) -> dict[str, Any]:
    systems = build_systems(emb, "eval")
    result = evaluate(systems, args.k, QUESTIONS, args.scope)
    result["environment"] = env_info(emb)
    result["config"] = {"k": args.k, "scope": args.scope, "model": args.emb, "device": args.device, "chunk_size": 120, "chunk_overlap": 20, "question_count": len(QUESTIONS), "seed": 0}
    return result


def print_result(result: dict[str, Any]) -> None:
    print(f"\n===== 文档级检索评测 Top-{result['k']} / Graph scope={result['graph_scope']} =====")
    print(f"{'范式':<14}{'类型':<10}{'Recall':>10}{'MRR':>10}{'完整命中':>10}")
    for name, val in result["systems"].items():
        for tp in ("fact", "multi", "global", "all"):
            g = val["groups"][tp]
            print(f"{name:<14}{tp:<10}{g['recall']:>10.3f}{g['mrr']:>10.3f}{g['full']:>7}/{g['n']:<3}")


def main() -> None:
    ap = argparse.ArgumentParser(description="作业四：三种 RAG 对比实验")
    ap.add_argument("--mode", choices=["eval", "vector", "graph", "wiki"], default="eval")
    ap.add_argument("--q", default="Q01")
    ap.add_argument("--scope", choices=["local", "global"], default="local")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--emb", default="paraphrase-multilingual-MiniLM-L12-v2", help="本地模型目录或 Hugging Face 模型名")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--mrr_check", default=None, help="如 3,1,2")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.mrr_check:
        ranks = [int(x) for x in args.mrr_check.split(",")]
        value = sum(1 / r for r in ranks) / len(ranks)
        print("MRR = (" + " + ".join(f"1/{r}" for r in ranks) + f") / {len(ranks)} = {value:.6f}")
        return

    t0 = time.time()
    emb = Embedder(args.emb, args.device)
    systems = build_systems(emb, args.mode)
    if args.mode == "eval":
        result = evaluate(systems, args.k, QUESTIONS, args.scope)
        result["environment"] = env_info(emb)
        result["config"] = {"k": args.k, "scope": args.scope, "model": args.emb, "device": args.device, "chunk_size": 120, "chunk_overlap": 20, "question_count": len(QUESTIONS), "seed": 0, "elapsed_seconds": round(time.time() - t0, 3)}
        print_result(result)
        if not args.no_save:
            save_json(outdir / f"eval_k{args.k}_{args.scope}.json", result)
            rows = []
            for name, val in result["systems"].items():
                for tp, g in val["groups"].items():
                    rows.append({"system": name, "type": tp, **g})
            write_csv(outdir / f"eval_k{args.k}_{args.scope}.csv", rows)
            save_json(outdir / f"retrieval_details_k{args.k}_{args.scope}.json", {n: v["rows"] for n, v in result["systems"].items()})
            plot_results(outdir, result, f"Recall@{args.k}（20题）")
        return

    q = next((q for q in QUESTIONS if q["id"].upper() == args.q.upper()), None)
    if q is None:
        raise SystemExit(f"未知题号：{args.q}")
    name = {"vector": "Vector RAG", "graph": "GraphRAG", "wiki": "WikiRAG"}[args.mode]
    system = systems[name]
    scored = system.scored(q["q"], args.scope) if name == "GraphRAG" else system.scored(q["q"])
    docs = [d for d, _, _ in scored[:args.k]]
    print(f"\n[{q['id']}·{q['type']}] {q['q']}\nevidence: {q['evidence']}\n检索到的文档：{docs}")
    for i, (doc, score, text) in enumerate(scored[:args.k], 1):
        print(f"  {i}. {doc} score={score:.4f} :: {text[:100]}")
    if not args.no_save:
        save_json(outdir / f"single_{args.mode}_{q['id']}.json", {"question": q, "docs": docs, "scored": scored, "environment": env_info(emb)})


if __name__ == "__main__":
    main()
