from __future__ import annotations

from app.schemas import DiagnosisResult


INTERNAL_TERMS = {
    "strong_root",
    "weak_root",
    "context_only",
    "candidate_contract",
    "best_hypothesis",
    "selection_class",
    "evidence_type",
    "candidate_dossiers",
    "root_selectable",
}

INTERNAL_PHRASES = {
    "graph ranking algorithm",
    "rank 1",
    "rank 2",
    "rank 3",
    "ranked first",
    "audited winner",
}


def diagnosis_text(result: DiagnosisResult) -> str:
    sections = [
        result.incident_summary,
        result.reasoning_summary,
        " ".join(result.recommended_remediation),
        " ".join(result.limitations),
    ]
    sections.extend(item.summary for item in result.evidence)
    return "\n".join(sections).lower()


def presentation_violations(result: DiagnosisResult) -> list[str]:
    text = diagnosis_text(result)
    violations = []

    for term in sorted(INTERNAL_TERMS):
        if term.lower() in text:
            violations.append(f"Final RCA exposes internal graph term: {term}")

    for phrase in sorted(INTERNAL_PHRASES):
        if phrase.lower() in text:
            violations.append(f"Final RCA exposes internal workflow phrase: {phrase}")

    if "score " in text or "score:" in text:
        violations.append("Final RCA exposes internal score values.")

    return violations
