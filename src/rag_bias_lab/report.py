"""Markdown report generation for a lab run."""

from datetime import date
from pathlib import Path


def _verdicts(res):
    pos, length, acc = res["position"], res["length"], res["accuracy"]
    out = []
    out.append(
        "PASS: no position bias detected"
        if pos["consistency"] >= 0.95 and abs(pos["first_position_rate"] - 0.5) <= 0.1
        else "FLAG: position bias detected"
    )
    out.append(
        "PASS: no length bias detected"
        if length["long_wrong_win_rate"] <= 0.1
        else "FLAG: length bias detected"
    )
    out.append(
        "PASS: judge is competent" if acc["accuracy"] >= 0.7 else "FLAG: judge accuracy is low"
    )
    return out


def write_report(path, config, retrieval_metrics, audit_results):
    lines = []
    lines.append("# rag-bias-lab run report\n")
    lines.append(f"Date: {date.today().isoformat()}\n")
    lines.append("## Config\n")
    for k, v in config.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## Retrieval + generation (SQuAD slice)\n")
    lines.append(f"- recall@3: {retrieval_metrics['recall_at_3']:.3f}")
    lines.append(f"- answer token-F1 (extractive): {retrieval_metrics['answer_f1']:.3f}")
    lines.append(f"- eval questions: {retrieval_metrics['n_eval']}")
    lines.append("\n## Judge bias audits\n")
    lines.append("| judge | position consistency | first-position rate | long-wrong win rate | accuracy |")
    lines.append("| --- | --- | --- | --- | --- |")
    for name, res in audit_results.items():
        p, l, a = res["position"], res["length"], res["accuracy"]
        lines.append(
            f"| {name} | {p['consistency']:.3f} (n={p['n']}) | {p['first_position_rate']:.3f} "
            f"| {l['long_wrong_win_rate']:.3f} (n={l['n']}) | {a['accuracy']:.3f} (n={a['n']}) |"
        )
    lines.append("\n## Verdicts\n")
    for name, res in audit_results.items():
        lines.append(f"### {name}")
        for v in _verdicts(res):
            lines.append(f"- {v}")
    lines.append(
        "\n## How to read this\n"
        "- *Position consistency* = fraction of pairs judged identically in both "
        "presentation orders. 1.0 means order never mattered.\n"
        "- *First-position rate* = how often the first-presented answer wins. "
        "0.5 means no positional preference.\n"
        "- *Long-wrong win rate* = how often a padded, verbose *wrong* answer beats "
        "the short correct one. Near 0.0 is unbiased.\n"
        "- The `position-biased (p=0.7)` and `length-biased (p=0.7)` judges are "
        "controls: they inject a *known* bias so you can verify the audit "
        "recovers it. If the audit can't flag a 70% injected bias, the audit "
        "itself is broken.\n"
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
