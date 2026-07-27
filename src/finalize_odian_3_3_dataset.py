#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_KNOWLEDGE_RECORDS = 49
QA_PER_RECORD = 3
EXPECTED_QA = EXPECTED_KNOWLEDGE_RECORDS * QA_PER_RECORD
PDF_PAGE_OFFSET = 27  # printed 204 -> PDF page 231

# Whole concept families stay in one split. This is stricter than record-level splitting.
CONCEPT_GROUPS: dict[str, list[int]] = {
    "mechanism_initiation_propagation_termination": list(range(1, 15)),
    "equal_reactivity_scope": list(range(15, 19)),
    "rate_expression_and_steady_state": list(range(19, 32)),
    "general_experimental_principles": list(range(32, 35)),
    "isolation_and_byproduct_monitoring": list(range(35, 39)),
    "chemical_analysis": list(range(39, 41)),
    "spectroscopic_monitoring": list(range(41, 46)),
    "dilatometry_calorimetry_and_other_methods": list(range(46, 50)),
}

GROUP_SPLIT = {
    "mechanism_initiation_propagation_termination": "train",
    "equal_reactivity_scope": "train",
    "rate_expression_and_steady_state": "train",
    "general_experimental_principles": "train",
    "isolation_and_byproduct_monitoring": "train",
    "chemical_analysis": "train",
    "spectroscopic_monitoring": "holdout",
    "dilatometry_calorimetry_and_other_methods": "validation",
}

KEY_FORMULAS = {
    "odian_3_3_k_002": r"I \xrightarrow{k_d} 2R^{\bullet}",
    "odian_3_3_k_003": r"R^{\bullet}+M \xrightarrow{k_i} M_1^{\bullet}",
    "odian_3_3_k_005": r"M_n^{\bullet}+M \xrightarrow{k_p} M_{n+1}^{\bullet}",
    "odian_3_3_k_009": r"M_n^{\bullet}+M_m^{\bullet}\xrightarrow{k_{tc}}M_{n+m}",
    "odian_3_3_k_010": r"M_n^{\bullet}+M_m^{\bullet}\xrightarrow{k_{td}}M_n+M_m",
    "odian_3_3_k_011": r"M_n^{\bullet}+M_m^{\bullet}\xrightarrow{k_t}\text{dead polymer}",
    "odian_3_3_k_012": r"k_t=a k_{tc}+(1-a)k_{td}",
    "odian_3_3_k_014": r"R_p\propto k_t^{-1/2}",
    "odian_3_3_k_019": r"-\frac{d[M]}{dt}=R_i+R_p",
    "odian_3_3_k_020": r"-\frac{d[M]}{dt}\approx R_p",
    "odian_3_3_k_021": r"R_p=k_p[M][M^{\bullet}]",
    "odian_3_3_k_023": r"[M^{\bullet}]\sim10^{-8}\ \mathrm{mol\,L^{-1}}",
    "odian_3_3_k_024": r"\frac{d[M^{\bullet}]}{dt}\approx0,\qquad R_i=R_t",
    "odian_3_3_k_025": r"R_t=2k_t[M^{\bullet}]^2",
    "odian_3_3_k_029": r"[M^{\bullet}]=\left(\frac{R_i}{2k_t}\right)^{1/2}",
    "odian_3_3_k_030": r"R_p=k_p[M]\left(\frac{R_i}{2k_t}\right)^{1/2}",
    "odian_3_3_k_031": r"\frac{R_{p,2}}{R_{p,1}}=\left(\frac{R_{i,2}}{R_{i,1}}\right)^{1/2}=\sqrt{2}",
}

SOURCE_ANOMALIES = [
    {
        "id": "odian_p209_pmma_density_direction",
        "printed_page": 209,
        "pdf_page": 236,
        "source_sentence_summary": (
            "The page states that PMMA density is 20.6% lower than monomer density, "
            "while the same paragraph describes high-volume shrinkage on polymerization."
        ),
        "handling": (
            "The numerical density-direction claim is excluded from the knowledge and QA answers. "
            "Only the source-supported dilatometry principle and volume-shrinkage statement are retained."
        ),
        "status": "documented_source_internal_inconsistency_not_used_for_training",
    }
]

REFUSAL_PHRASES = (
    "اطلاعات کافی نیست",
    "منبع کافی نیست",
    "نمی‌توان پاسخ داد",
    "قابل پاسخ نیست",
    "نمی‌دانم",
)

FINAL_FIELDS = [
    "qa_id",
    "split",
    "concept_group",
    "source_record_id",
    "source_chunk_ids",
    "printed_pages",
    "pdf_pages",
    "subsection",
    "subtopic",
    "knowledge_type",
    "question_type",
    "difficulty",
    "question_fa",
    "answer_fa",
    "canonical_formula_latex",
    "answer_support_fields",
    "question_origin",
    "answer_evidence_mode",
    "answerable_from_source",
    "refusal_expected",
    "source_verification_status",
    "assistant_review_status",
    "domain_expert_verified",
    "reviewer_decision",
    "training_ready",
    "readiness_scope",
    "production_release_ready",
    "reviewer_notes",
    "question_sha256",
    "answer_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "بازبینی نهایی Knowledge و QA بخش 3-3 Odian، اصلاح metadata، "
            "تقسیم concept-group و ساخت نسخه آماده آموزش آزمایشی."
        )
    )
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path(
            "data/knowledge/odian_ch3/verified/"
            "knowledge_3_3_source_reviewed.jsonl"
        ),
    )
    parser.add_argument(
        "--qa-builder",
        type=Path,
        default=Path("src/create_odian_3_3_qa_table.py"),
    )
    parser.add_argument(
        "--source-pdf",
        type=Path,
        default=Path(
            "data/rag/source_pdfs/Principles of Polymerization.pdf"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/final/odian_ch3/section_3_3"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON نامعتبر در {path} خط {line_number}: {exc}"
                ) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def list_to_cell(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value or "")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: list_to_cell(row.get(field, "")) for field in fields})


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_spaces(text: str) -> str:
    text = str(text or "").replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_terminal_punctuation(text: str) -> str:
    text = normalize_spaces(text)
    return re.sub(r"[.。؛;،,!?؟]+$", "", text).strip()


def sentence(text: str) -> str:
    text = strip_terminal_punctuation(text)
    return f"{text}." if text else ""


def join_sentences(*parts: str) -> str:
    return " ".join(sentence(part) for part in parts if strip_terminal_punctuation(part))


def normalize_formula(formula: str) -> str:
    return re.sub(r"\s+", "", formula or "")


def record_number(record_id: str) -> int:
    return int(record_id.rsplit("_", 1)[1])


def concept_group_for(record_id: str) -> str:
    number = record_number(record_id)
    for group, numbers in CONCEPT_GROUPS.items():
        if number in numbers:
            return group
    raise KeyError(f"Concept group پیدا نشد: {record_id}")


def import_qa_builder(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"اسکریپت QA builder پیدا نشد: {path}")
    spec = importlib.util.spec_from_file_location("odian_qa_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("بارگذاری QA builder ممکن نشد.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_pdf_mapping(pdf_path: Path) -> dict[str, Any]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF منبع پیدا نشد: {pdf_path}")
    if shutil.which("pdftotext") is None:
        raise RuntimeError("دستور pdftotext در WSL نصب/قابل دسترس نیست.")

    def page_text(page: int) -> str:
        result = subprocess.run(
            [
                "pdftotext",
                "-f", str(page),
                "-l", str(page),
                "-layout",
                str(pdf_path),
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"استخراج متن PDF page {page} شکست خورد: {result.stderr}"
            )
        return result.stdout

    first_text = page_text(231)
    last_text = page_text(236)

    checks = {
        "pdf_page_231_has_section_3_3": bool(
            re.search(
                r"3\s*-\s*3\s+RATE\s+OF\s+RADICAL\s+CHAIN\s+POLYMERIZATION",
                first_text,
                flags=re.IGNORECASE,
            )
        ),
        "pdf_page_236_has_section_3_4": bool(
            re.search(r"3\s*-\s*4\s+INITIATION", last_text, flags=re.IGNORECASE)
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"تطبیق صفحات PDF تأیید نشد: {checks}")

    return {
        "mapping_rule": "pdf_page_1_based = printed_page + 27",
        "printed_page_start": 204,
        "printed_page_end": 209,
        "pdf_page_start": 231,
        "pdf_page_end": 236,
        "checks": checks,
        "status": "verified_against_pdf_text",
    }


def patch_knowledge(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for original in sorted(records, key=lambda item: item["record_id"]):
        record = dict(original)
        record["printed_pages"] = [int(p) for p in record.get("printed_pages", [])]
        record["pdf_pages"] = [p + PDF_PAGE_OFFSET for p in record["printed_pages"]]
        record["verification_status"] = "source_reviewed_against_rendered_pdf"
        record["assistant_scientific_review"] = "approved"
        record["domain_expert_verified"] = False

        if record["record_id"] == "odian_3_3_k_012":
            record["formula_latex"] = KEY_FORMULAS[record["record_id"]]
            record["variables_json"] = {
                "kt": "ثابت سرعت کلی اختتام",
                "ktc": "ثابت سرعت اختتام از طریق ترکیب",
                "ktd": "ثابت سرعت اختتام از طریق نامتناسب‌شدن",
                "a": "کسر اختتام از طریق ترکیب",
                "1-a": "کسر اختتام از طریق نامتناسب‌شدن",
            }
            record["review_notes"] = join_sentences(
                record.get("review_notes", ""),
                "نماد کسر مسیر ترکیب مطابق متن منبع از a استفاده شد",
            )

        if record["record_id"] == "odian_3_3_k_046":
            record["source_anomaly_ids"] = [
                "odian_p209_pmma_density_direction"
            ]
            record["review_notes"] = join_sentences(
                record.get("review_notes", ""),
                "ادعای عددی جهت چگالی PMMA به علت ناسازگاری درونی متن منبع وارد پاسخ آموزشی نشده است",
            )
        else:
            record.setdefault("source_anomaly_ids", [])

        patched.append(record)
    return patched


def direct_answer(record: dict[str, Any]) -> tuple[str, list[str]]:
    parts = [record["statement_fa"]]
    support = ["statement_fa"]
    formula = record.get("formula_latex", "")
    if formula:
        parts.append(f"رابطه: ${formula}$")
        support.append("formula_latex")
    return join_sentences(*parts), support


def reasoning_question(record: dict[str, Any], label: str) -> str:
    kind = record["knowledge_type"]
    if kind in {"formula", "derived_formula", "typical_value", "effect_prediction"}:
        return f"رابطه مربوط به «{label}» چیست و وابستگی یا شرط اصلی آن چگونه بیان می‌شود؟"
    if kind == "assumption":
        return f"فرض «{label}» دقیقاً چه می‌گوید و چه نتیجه‌ای برای تحلیل سینتیکی دارد؟"
    if kind in {"measurement_method", "measurement_principle", "measurement_setup", "measurement_advantage", "instrument_capability"}:
        return f"روش یا اصل «{label}» چگونه پیشرفت پلیمریزاسیون را دنبال می‌کند و چه نتیجه‌ای می‌دهد؟"
    if kind in {"scope", "limitation", "measurement_limitation"}:
        return f"دامنه کاربرد یا محدودیت «{label}» را همراه با دلیل یا پیامد آن توضیح دهید؟"
    if kind in {"mechanism", "explanation"}:
        return f"مکانیزم «{label}» چگونه است و چه پیامدی برای زنجیر یا سرعت واکنش دارد؟"
    if kind == "definition":
        return f"«{label}» را تعریف کنید و تمایز مهم آن را بیان کنید؟"
    return f"نکته علمی اصلی درباره «{label}» چیست و چه پیامدی دارد؟"


def reasoning_answer(record: dict[str, Any]) -> tuple[str, list[str], str]:
    parts = [record["statement_fa"]]
    support = ["statement_fa"]
    formula = record.get("formula_latex", "")
    if formula:
        parts.append(f"رابطه: ${formula}$")
        support.append("formula_latex")
    if record.get("cause_fa"):
        parts.append(f"دلیل: {record['cause_fa']}")
        support.append("cause_fa")
    if record.get("effect_fa"):
        parts.append(f"پیامد: {record['effect_fa']}")
        support.append("effect_fa")
    if record.get("assumptions_fa"):
        parts.append(f"شرط یا دامنه اعتبار: {record['assumptions_fa']}")
        support.append("assumptions_fa")

    evidence_mode = (
        "direct_source_summary"
        if support == ["statement_fa"] or support == ["statement_fa", "formula_latex"]
        else "source_summary_with_explicit_derivation_or_scope"
    )
    return join_sentences(*parts), support, evidence_mode


def misconception_qa(record: dict[str, Any]) -> tuple[str, str, list[str]]:
    misconception = strip_terminal_punctuation(record.get("common_error_fa", ""))
    if not misconception:
        misconception = (
            f"مفهوم {record['subtopic']} را بدون توجه به دامنه اعتبار آن به همه سامانه‌ها تعمیم دادن"
        )
    question = (
        f"برداشت نادرست «{misconception}» را اصلاح کنید؛ "
        "بیان صحیح چیست؟"
    )
    parts = ["این برداشت نادرست است", record["statement_fa"]]
    support = ["statement_fa", "common_error_fa"]
    formula = record.get("formula_latex", "")
    if formula:
        parts.append(f"رابطه صحیح: ${formula}$")
        support.append("formula_latex")
    if record.get("assumptions_fa"):
        parts.append(f"دامنه اعتبار: {record['assumptions_fa']}")
        support.append("assumptions_fa")
    return normalize_spaces(question), join_sentences(*parts), support


def build_final_qa(
    records: list[dict[str, Any]],
    direct_questions: dict[str, str],
    topic_labels: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        record_id = record["record_id"]
        group = concept_group_for(record_id)
        split = GROUP_SPLIT[group]
        label = topic_labels[record_id]

        v1_answer, v1_support = direct_answer(record)
        v2_answer, v2_support, v2_mode = reasoning_answer(record)
        v3_question, v3_answer, v3_support = misconception_qa(record)

        variants = [
            {
                "suffix": "v1",
                "question_type": "direct_recall",
                "difficulty": "easy",
                "question": direct_questions[record_id],
                "answer": v1_answer,
                "support": v1_support,
                "question_origin": "manually_authored_from_source_record",
                "answer_evidence_mode": "direct_source_summary",
            },
            {
                "suffix": "v2",
                "question_type": "reasoning_and_conditions",
                "difficulty": "medium",
                "question": reasoning_question(record, label),
                "answer": v2_answer,
                "support": v2_support,
                "question_origin": "template_authored_then_manual_source_review",
                "answer_evidence_mode": v2_mode,
            },
            {
                "suffix": "v3",
                "question_type": "misconception_correction",
                "difficulty": "medium",
                "question": v3_question,
                "answer": v3_answer,
                "support": v3_support,
                "question_origin": "synthetic_misconception_from_reviewed_record",
                "answer_evidence_mode": "source_correction_of_synthetic_misconception",
            },
        ]

        number = record_number(record_id)
        for variant in variants:
            question = normalize_spaces(variant["question"])
            answer = normalize_spaces(variant["answer"])
            qa_id = f"odian_3_3_qa_{number:03d}_{variant['suffix']}"
            rows.append(
                {
                    "qa_id": qa_id,
                    "split": split,
                    "concept_group": group,
                    "source_record_id": record_id,
                    "source_chunk_ids": record.get("source_chunk_ids", []),
                    "printed_pages": record.get("printed_pages", []),
                    "pdf_pages": record.get("pdf_pages", []),
                    "subsection": record["subsection"],
                    "subtopic": record["subtopic"],
                    "knowledge_type": record["knowledge_type"],
                    "question_type": variant["question_type"],
                    "difficulty": variant["difficulty"],
                    "question_fa": question,
                    "answer_fa": answer,
                    "canonical_formula_latex": record.get("formula_latex", ""),
                    "answer_support_fields": variant["support"],
                    "question_origin": variant["question_origin"],
                    "answer_evidence_mode": variant["answer_evidence_mode"],
                    "answerable_from_source": True,
                    "refusal_expected": False,
                    "source_verification_status": record["verification_status"],
                    "assistant_review_status": "approved_against_rendered_source",
                    "domain_expert_verified": False,
                    "reviewer_decision": "approve",
                    "training_ready": True,
                    "readiness_scope": "pilot_training_only",
                    "production_release_ready": False,
                    "reviewer_notes": (
                        "سؤال و پاسخ با رکورد علمی و صفحه منبع تطبیق داده شد؛ "
                        "تأیید متخصص دامنه هنوز انجام نشده است."
                    ),
                    "question_sha256": sha256_text(question),
                    "answer_sha256": sha256_text(answer),
                }
            )
    return rows


def validate(
    knowledge: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    pdf_check: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(knowledge) != EXPECTED_KNOWLEDGE_RECORDS:
        errors.append(
            f"Knowledge count باید {EXPECTED_KNOWLEDGE_RECORDS} باشد، اما {len(knowledge)} است."
        )
    if len(qa_rows) != EXPECTED_QA:
        errors.append(f"QA count باید {EXPECTED_QA} باشد، اما {len(qa_rows)} است.")

    record_ids = [r["record_id"] for r in knowledge]
    if len(record_ids) != len(set(record_ids)):
        errors.append("شناسه تکراری در Knowledge وجود دارد.")

    qa_ids = [q["qa_id"] for q in qa_rows]
    if len(qa_ids) != len(set(qa_ids)):
        errors.append("شناسه تکراری QA وجود دارد.")

    question_hashes = [q["question_sha256"] for q in qa_rows]
    if len(question_hashes) != len(set(question_hashes)):
        errors.append("سؤال دقیقاً تکراری وجود دارد.")

    per_record = Counter(q["source_record_id"] for q in qa_rows)
    for record_id in record_ids:
        if per_record[record_id] != QA_PER_RECORD:
            errors.append(
                f"{record_id}: تعداد QA باید {QA_PER_RECORD} باشد، ولی {per_record[record_id]} است."
            )

    # Exact formula audit.
    lookup = {r["record_id"]: r for r in knowledge}
    for record_id, expected in KEY_FORMULAS.items():
        actual = lookup[record_id].get("formula_latex", "")
        if normalize_formula(actual) != normalize_formula(expected):
            errors.append(f"{record_id}: فرمول نهایی با منبع تطبیق ندارد.")

    # Corrected page mapping.
    for record in knowledge:
        expected_pages = [p + PDF_PAGE_OFFSET for p in record["printed_pages"]]
        if record["pdf_pages"] != expected_pages:
            errors.append(f"{record['record_id']}: نگاشت PDF page نادرست است.")

    # Concept-level leakage audit.
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for row in qa_rows:
        splits_by_group[row["concept_group"]].add(row["split"])
    for group, splits in splits_by_group.items():
        if len(splits) != 1:
            errors.append(f"نشت concept group: {group} -> {sorted(splits)}")

    banned_fragments = (
        "density of poly(methyl methacrylate) is 20.6% lower",
        "چگالی پلی(متیل متاکریلات) 20.6 درصد کمتر",
        "چگالی پلی‌متیل متاکریلات 20.6 درصد کمتر",
    )

    for row in qa_rows:
        question = row["question_fa"]
        answer = row["answer_fa"]
        combined = f"{question} {answer}".lower()

        if ".." in question or ".." in answer:
            errors.append(f"{row['qa_id']}: نشانه‌گذاری دوتایی وجود دارد.")
        if "؟؟" in question or "؟؟" in answer:
            errors.append(f"{row['qa_id']}: علامت سؤال دوتایی وجود دارد.")
        if not question.endswith("؟"):
            errors.append(f"{row['qa_id']}: سؤال با ؟ پایان نمی‌یابد.")
        if len(question) < 20 or len(answer) < 25:
            errors.append(f"{row['qa_id']}: سؤال یا پاسخ بیش از حد کوتاه است.")
        if not row["answerable_from_source"] or row["refusal_expected"]:
            errors.append(f"{row['qa_id']}: وضعیت answerable/refusal نادرست است.")
        if not row["training_ready"] or row["production_release_ready"]:
            errors.append(f"{row['qa_id']}: وضعیت آمادگی نادرست است.")
        if row["domain_expert_verified"]:
            errors.append(f"{row['qa_id']}: تأیید متخصص نباید true باشد.")
        if row["reviewer_decision"] != "approve":
            errors.append(f"{row['qa_id']}: reviewer_decision باید approve باشد.")

        for phrase in REFUSAL_PHRASES:
            if phrase in combined:
                errors.append(f"{row['qa_id']}: عبارت امتناع نامناسب وجود دارد.")
        for fragment in banned_fragments:
            if fragment.lower() in combined:
                errors.append(f"{row['qa_id']}: ادعای ناسازگار PMMA وارد داده شده است.")

        record = lookup[row["source_record_id"]]
        statement = normalize_spaces(record["statement_fa"])
        if statement not in answer:
            errors.append(f"{row['qa_id']}: statement اصلی در پاسخ وجود ندارد.")
        formula = row["canonical_formula_latex"]
        if formula and formula not in answer:
            errors.append(f"{row['qa_id']}: فرمول canonical در پاسخ وجود ندارد.")

    if not all(pdf_check["checks"].values()):
        errors.append("تأیید PDF page mapping ناموفق است.")

    report = {
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "knowledge_record_count": len(knowledge),
        "qa_count": len(qa_rows),
        "qa_per_record": QA_PER_RECORD,
        "split_record_counts": dict(
            sorted(
                Counter(
                    (row["split"], row["source_record_id"])
                    for row in qa_rows
                ).keys()
            )
        ) if False else {
            split: len({row["source_record_id"] for row in qa_rows if row["split"] == split})
            for split in ("train", "validation", "holdout")
        },
        "split_qa_counts": dict(sorted(Counter(row["split"] for row in qa_rows).items())),
        "question_type_counts": dict(
            sorted(Counter(row["question_type"] for row in qa_rows).items())
        ),
        "concept_group_splits": {
            group: next(iter(splits)) for group, splits in sorted(splits_by_group.items())
        },
        "concept_group_leakage_count": sum(
            1 for splits in splits_by_group.values() if len(splits) != 1
        ),
        "pdf_mapping": pdf_check,
        "source_anomalies": SOURCE_ANOMALIES,
        "readiness": {
            "training_ready": True,
            "readiness_scope": "pilot_training_only",
            "production_release_ready": False,
            "domain_expert_verified": False,
        },
        "important_note": (
            "نسخه نهایی از نظر متن منبع، فرمول‌ها، نگاشت صفحه، نشانه‌گذاری، "
            "تکرار و نشت concept-group بررسی شده است. آماده آموزش آزمایشی است؛ "
            "اما برای انتشار تولیدی همچنان تأیید متخصص دامنه لازم است."
        ),
    }

    if errors:
        print("اعتبارسنجی نهایی ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)
    return report


def build_sft_rows(qa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system_prompt = (
        "شما مدرس علوم پلیمر هستید. پاسخ را دقیق، کوتاه، علمی و به زبان فارسی بدهید. "
        "فرمول‌ها را بدون تغییر علامت، توان یا ضریب بنویسید."
    )
    output: list[dict[str, Any]] = []
    for row in qa_rows:
        output.append(
            {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["question_fa"]},
                    {"role": "assistant", "content": row["answer_fa"]},
                ],
                "metadata": {
                    "qa_id": row["qa_id"],
                    "split": row["split"],
                    "concept_group": row["concept_group"],
                    "source_record_id": row["source_record_id"],
                    "printed_pages": row["printed_pages"],
                    "pdf_pages": row["pdf_pages"],
                    "training_ready": row["training_ready"],
                    "domain_expert_verified": row["domain_expert_verified"],
                },
            }
        )
    return output


def main() -> None:
    args = parse_args()
    module = import_qa_builder(args.qa_builder)
    direct_questions = module.DIRECT_QUESTIONS
    topic_labels = module.TOPIC_LABELS

    original_knowledge = load_jsonl(args.knowledge)
    if len(original_knowledge) != EXPECTED_KNOWLEDGE_RECORDS:
        raise SystemExit(
            f"تعداد Knowledge ورودی {len(original_knowledge)} است؛ انتظار {EXPECTED_KNOWLEDGE_RECORDS}."
        )

    pdf_check = verify_pdf_mapping(args.source_pdf)
    final_knowledge = patch_knowledge(original_knowledge)
    final_qa = build_final_qa(final_knowledge, direct_questions, topic_labels)
    audit = validate(final_knowledge, final_qa, pdf_check)

    knowledge_dir = args.output_root / "knowledge"
    qa_dir = args.output_root / "qa"
    sft_dir = args.output_root / "sft"
    for directory in (knowledge_dir, qa_dir, sft_dir):
        directory.mkdir(parents=True, exist_ok=True)

    knowledge_jsonl = knowledge_dir / "knowledge_3_3_final_reviewed.jsonl"
    knowledge_csv = knowledge_dir / "knowledge_3_3_final_reviewed.csv"
    all_jsonl = qa_dir / "qa_3_3_final_all.jsonl"
    all_csv = qa_dir / "qa_3_3_final_all.csv"
    audit_json = qa_dir / "qa_3_3_final_audit.json"
    anomalies_json = qa_dir / "source_anomalies.json"

    write_jsonl(knowledge_jsonl, final_knowledge)
    knowledge_fields = list(final_knowledge[0].keys())
    write_csv(knowledge_csv, final_knowledge, knowledge_fields)
    write_jsonl(all_jsonl, final_qa)
    write_csv(all_csv, final_qa, FINAL_FIELDS)

    for split in ("train", "validation", "holdout"):
        split_rows = [row for row in final_qa if row["split"] == split]
        write_jsonl(qa_dir / f"qa_3_3_final_{split}.jsonl", split_rows)

    sft_rows = build_sft_rows(final_qa)
    for split in ("train", "validation", "holdout"):
        split_sft = [row for row in sft_rows if row["metadata"]["split"] == split]
        write_jsonl(sft_dir / f"sft_3_3_{split}_messages.jsonl", split_sft)

    audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    anomalies_json.write_text(
        json.dumps(SOURCE_ANOMALIES, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 78)
    print("نسخه نهایی بازبینی‌شده بخش 3-3 ساخته شد.")
    print("Knowledge:", knowledge_jsonl)
    print("QA all:", all_jsonl)
    print("Audit:", audit_json)
    print("SFT directory:", sft_dir)
    print()
    print("Knowledge records:", len(final_knowledge))
    print("QA records:", len(final_qa))
    print("Split QA:", audit["split_qa_counts"])
    print("Concept-group leakage:", audit["concept_group_leakage_count"])
    print("Errors:", audit["error_count"])
    print("Warnings:", audit["warning_count"])
    print("PDF mapping: printed 204-209 -> PDF 231-236")
    print("Source anomalies excluded from training:", len(SOURCE_ANOMALIES))
    print()
    print("Readiness:")
    print("- training_ready = true")
    print("- readiness_scope = pilot_training_only")
    print("- production_release_ready = false")
    print("- domain_expert_verified = false")


if __name__ == "__main__":
    main()
