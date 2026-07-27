#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(
    "data/knowledge/odian_ch3/verified/"
    "knowledge_3_3_source_reviewed.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "data/qa/odian_ch3/section_3_3"
)

EXPECTED_RECORD_COUNT = 49
QA_PER_RECORD = 3
EXPECTED_QA_COUNT = EXPECTED_RECORD_COUNT * QA_PER_RECORD

VALIDATION_RECORD_IDS = {
    "odian_3_3_k_007",
    "odian_3_3_k_012",
    "odian_3_3_k_019",
    "odian_3_3_k_025",
    "odian_3_3_k_032",
    "odian_3_3_k_037",
    "odian_3_3_k_039",
    "odian_3_3_k_048",
}

HOLDOUT_RECORD_IDS = {
    "odian_3_3_k_003",
    "odian_3_3_k_010",
    "odian_3_3_k_020",
    "odian_3_3_k_024",
    "odian_3_3_k_030",
    "odian_3_3_k_033",
    "odian_3_3_k_036",
    "odian_3_3_k_043",
    "odian_3_3_k_046",
}

TOPIC_LABELS = {
    "odian_3_3_k_001": "توالی مراحل پلیمریزاسیون زنجیره‌ای رادیکالی",
    "odian_3_3_k_002": "تفکیک آغازگر",
    "odian_3_3_k_003": "تشکیل رادیکال آغازکننده زنجیر",
    "odian_3_3_k_004": "تفاوت رادیکال اولیه و رادیکال زنجیری",
    "odian_3_3_k_005": "واکنش رشد زنجیر",
    "odian_3_3_k_006": "بقای مرکز فعال در رشد",
    "odian_3_3_k_007": "محدوده معمول ثابت سرعت رشد",
    "odian_3_3_k_008": "اختتام دومولکولی",
    "odian_3_3_k_009": "اختتام از طریق ترکیب",
    "odian_3_3_k_010": "اختتام از طریق نامتناسب‌شدن",
    "odian_3_3_k_011": "نمایش کلی واکنش اختتام",
    "odian_3_3_k_012": "ثابت سرعت کلی اختتام",
    "odian_3_3_k_013": "پلیمر مرده",
    "odian_3_3_k_014": "مقایسه رشد و اختتام",
    "odian_3_3_k_015": "دامنه سازوکار روابط 3-13 تا 3-19",
    "odian_3_3_k_016": "فرض واکنش‌پذیری برابر",
    "odian_3_3_k_017": "اثر اندازه رادیکال‌های کوچک",
    "odian_3_3_k_018": "محدودیت فرض واکنش‌پذیری برابر",
    "odian_3_3_k_019": "سرعت کل ناپدیدشدن مونومر",
    "odian_3_3_k_020": "تقریب سرعت پلیمریزاسیون با سرعت رشد",
    "odian_3_3_k_021": "رابطه سرعت رشد",
    "odian_3_3_k_022": "غلظت کل رادیکال‌های زنجیری",
    "odian_3_3_k_023": "مرتبه غلظت رادیکال‌ها",
    "odian_3_3_k_024": "فرض حالت پایا",
    "odian_3_3_k_025": "رابطه سرعت اختتام",
    "odian_3_3_k_026": "زمان رسیدن به حالت پایا",
    "odian_3_3_k_027": "یکسانی فرم سینتیکی مسیرهای اختتام",
    "odian_3_3_k_028": "قرارداد ضریب دو",
    "odian_3_3_k_029": "غلظت حالت پایای رادیکال‌ها",
    "odian_3_3_k_030": "رابطه نهایی سرعت پلیمریزاسیون",
    "odian_3_3_k_031": "وابستگی ریشه دوم سرعت به آغازش",
    "odian_3_3_k_032": "اصل تجربی تعیین سرعت پلیمریزاسیون",
    "odian_3_3_k_033": "انتخاب روش اندازه‌گیری",
    "odian_3_3_k_034": "پایش پیوسته تبدیل",
    "odian_3_3_k_035": "روش جداسازی و وزن‌کردن پلیمر",
    "odian_3_3_k_036": "دامنه کاربرد روش جداسازی",
    "odian_3_3_k_037": "محدودیت‌های روش جداسازی",
    "odian_3_3_k_038": "پایش محصول جانبی در پلیمریزاسیون مرحله‌ای",
    "odian_3_3_k_039": "تحلیل گروه‌های عاملی",
    "odian_3_3_k_040": "تیتر برم برای مونومرهای وینیلی",
    "odian_3_3_k_041": "پایش طیف‌سنجی پلیمریزاسیون",
    "odian_3_3_k_042": "مثال NMR پلیمریزاسیون استایرن",
    "odian_3_3_k_043": "دقت و پیوستگی روش‌های طیفی",
    "odian_3_3_k_044": "آرایش پایش درجا در طیف‌سنج",
    "odian_3_3_k_045": "اندازه‌گیری هم‌زمان تبدیل و وزن مولکولی",
    "odian_3_3_k_046": "دیلاتومتری",
    "odian_3_3_k_047": "محدودیت دیلاتومتری",
    "odian_3_3_k_048": "تعیین تبدیل با DSC",
    "odian_3_3_k_049": "پراکندگی نور و ضریب شکست",
}

DIRECT_QUESTIONS = {
    "odian_3_3_k_001": "سه مرحله اصلی پلیمریزاسیون زنجیره‌ای رادیکالی کدام‌اند؟",
    "odian_3_3_k_002": "در بخش نخست آغازش، گونه آغازگر چگونه رادیکال اولیه تولید می‌کند؟",
    "odian_3_3_k_003": "رادیکال آغازکننده زنجیر چگونه از واکنش رادیکال اولیه و مونومر تشکیل می‌شود؟",
    "odian_3_3_k_004": "تفاوت R• با M1• چیست؟",
    "odian_3_3_k_005": "گام عمومی رشد زنجیر در پلیمریزاسیون رادیکالی چگونه نوشته می‌شود؟",
    "odian_3_3_k_006": "چرا زنجیر رادیکالی پس از هر گام رشد می‌تواند همچنان به رشد ادامه دهد؟",
    "odian_3_3_k_007": "محدوده تقریبی kp برای بسیاری از مونومرها چقدر است؟",
    "odian_3_3_k_008": "اختتام دومولکولی در پلیمریزاسیون رادیکالی چگونه رخ می‌دهد؟",
    "odian_3_3_k_009": "در اختتام از نوع ترکیب چه محصولی تشکیل می‌شود؟",
    "odian_3_3_k_010": "در اختتام از نوع نامتناسب‌شدن چه اتفاقی می‌افتد؟",
    "odian_3_3_k_011": "وقتی مسیر دقیق اختتام معلوم نباشد، واکنش اختتام چگونه به‌صورت کلی نمایش داده می‌شود؟",
    "odian_3_3_k_012": "رابطه ثابت سرعت کلی اختتام بر حسب ktc و ktd چیست؟",
    "odian_3_3_k_013": "منظور از پلیمر مرده در پلیمریزاسیون رادیکالی چیست؟",
    "odian_3_3_k_014": "چرا با وجود بزرگ‌تر بودن kt نسبت به kp، رشد زنجیر همچنان ادامه می‌یابد؟",
    "odian_3_3_k_015": "روابط 3-13 تا 3-19 چه بخشی از پلیمریزاسیون زنجیره‌ای رادیکالی را توصیف می‌کنند؟",
    "odian_3_3_k_016": "فرض واکنش‌پذیری برابر در استخراج رابطه سرعت چه می‌گوید؟",
    "odian_3_3_k_017": "اثر اندازه زنجیر بر واکنش‌پذیری رادیکال‌های بسیار کوچک چگونه است؟",
    "odian_3_3_k_018": "آیا فرض واکنش‌پذیری برابر یک قانون دقیق و بدون استثنا است؟",
    "odian_3_3_k_019": "رابطه سرعت کل ناپدیدشدن مونومر چیست؟",
    "odian_3_3_k_020": "در تولید پلیمر پرجرم، چرا سرعت پلیمریزاسیون را تقریباً برابر Rp می‌گیرند؟",
    "odian_3_3_k_021": "رابطه سرعت رشد بر حسب kp، غلظت مونومر و غلظت رادیکال‌ها چیست؟",
    "odian_3_3_k_022": "نماد [M•] در رابطه سرعت رشد به چه معناست؟",
    "odian_3_3_k_023": "غلظت معمول رادیکال‌های زنجیری تقریباً در چه مرتبه‌ای است و چرا اندازه‌گیری مستقیم آن دشوار است؟",
    "odian_3_3_k_024": "فرض حالت پایا برای غلظت رادیکال‌ها چه بیان می‌کند؟",
    "odian_3_3_k_025": "رابطه سرعت نابودی رادیکال‌ها در اختتام دومولکولی چیست و ضریب 2 از کجا می‌آید؟",
    "odian_3_3_k_026": "پلیمریزاسیون‌های معمول تقریباً پس از چه مدت می‌توانند به حالت پایا برسند؟",
    "odian_3_3_k_027": "چرا برای نوشتن رابطه کلی سرعت اختتام لازم نیست مسیر ترکیب یا نامتناسب‌شدن مشخص باشد؟",
    "odian_3_3_k_028": "قرارداد ضریب 2 در تعریف سرعت تولید یا نابودی رادیکال‌ها چیست؟",
    "odian_3_3_k_029": "غلظت حالت پایای رادیکال‌های زنجیری چگونه بر حسب Ri و kt بیان می‌شود؟",
    "odian_3_3_k_030": "رابطه نهایی سرعت پلیمریزاسیون رادیکالی در حالت پایا چیست؟",
    "odian_3_3_k_031": "اگر Ri دو برابر شود و سایر شرایط ثابت بمانند، Rp چند برابر می‌شود؟",
    "odian_3_3_k_032": "اصل کلی تعیین تجربی سرعت پلیمریزاسیون چیست؟",
    "odian_3_3_k_033": "در انتخاب روش اندازه‌گیری سرعت پلیمریزاسیون چه عواملی باید در نظر گرفته شوند؟",
    "odian_3_3_k_034": "پایش پیوسته تبدیل چه تفاوتی با نمونه‌برداری جداگانه دارد؟",
    "odian_3_3_k_035": "روش جداسازی فیزیکی و وزن‌کردن پلیمر چگونه برای دنبال‌کردن واکنش استفاده می‌شود؟",
    "odian_3_3_k_036": "چرا روش جداسازی و وزن‌کردن بیشتر برای پلیمریزاسیون زنجیره‌ای مناسب است؟",
    "odian_3_3_k_037": "محدودیت‌های اصلی روش جداسازی، خشک‌کردن و وزن‌کردن چیست؟",
    "odian_3_3_k_038": "چگونه می‌توان پیشرفت یک پلیمریزاسیون مرحله‌ای را با پایش محصول جانبی دنبال کرد؟",
    "odian_3_3_k_039": "تحلیل گروه‌های عاملی چگونه برای تعیین سرعت پلیمریزاسیون مرحله‌ای به‌کار می‌رود؟",
    "odian_3_3_k_040": "تیتر برم چگونه برای دنبال‌کردن پلیمریزاسیون مونومرهای وینیلی استفاده می‌شود؟",
    "odian_3_3_k_041": "روش‌های طیف‌سنجی چگونه ناپدیدشدن مونومر یا پیدایش پلیمر را دنبال می‌کنند؟",
    "odian_3_3_k_042": "در مثال NMR پلیمریزاسیون استایرن، کدام سیگنال‌ها با تبدیل کاهش یا ظاهر می‌شوند؟",
    "odian_3_3_k_043": "چه شرایطی دقت پایش طیف‌سنجی پلیمریزاسیون را افزایش می‌دهد؟",
    "odian_3_3_k_044": "پایش درجا داخل طیف‌سنج چگونه انجام می‌شود؟",
    "odian_3_3_k_045": "ابزارهای پیشرفته‌تر چه دو کمیتی را می‌توانند هم‌زمان در طول پلیمریزاسیون اندازه‌گیری کنند؟",
    "odian_3_3_k_046": "دیلاتومتری بر چه اصلی استوار است و برای چه نوع سامانه‌ای مناسب است؟",
    "odian_3_3_k_047": "چرا دیلاتومتری معمولاً برای پلیمریزاسیون مرحله‌ای دارای محصول جانبی کوچک مناسب نیست؟",
    "odian_3_3_k_048": "چگونه می‌توان با DSC تبدیل پلیمریزاسیون را تعیین کرد؟",
    "odian_3_3_k_049": "غیر از روش‌های شیمیایی، طیفی و دیلاتومتری، چه روش‌های دیگری برای دنبال‌کردن پلیمریزاسیون ذکر شده‌اند؟",
}

KNOWN_FORMULAS = {
    "odian_3_3_k_019": r"-\frac{d[M]}{dt}=R_i+R_p",
    "odian_3_3_k_020": r"-\frac{d[M]}{dt}\approx R_p",
    "odian_3_3_k_021": r"R_p=k_p[M][M^{\bullet}]",
    "odian_3_3_k_024": r"\frac{d[M^{\bullet}]}{dt}\approx0,\qquad R_i=R_t",
    "odian_3_3_k_025": r"R_t=2k_t[M^{\bullet}]^2",
    "odian_3_3_k_029": r"[M^{\bullet}]=\left(\frac{R_i}{2k_t}\right)^{1/2}",
    "odian_3_3_k_030": r"R_p=k_p[M]\left(\frac{R_i}{2k_t}\right)^{1/2}",
    "odian_3_3_k_031": r"\frac{R_{p,2}}{R_{p,1}}=\left(\frac{R_{i,2}}{R_{i,1}}\right)^{1/2}=\sqrt{2}",
}

REFUSAL_PHRASES = (
    "اطلاعات کافی نیست",
    "منبع کافی نیست",
    "نمی‌توان پاسخ داد",
    "قابل پاسخ نیست",
    "نمی‌دانم",
)

CSV_FIELDS = [
    "qa_id",
    "split",
    "leakage_group",
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
    "assumptions_fa",
    "evidence_statement_fa",
    "answer_support_fields",
    "answerable_from_source",
    "refusal_expected",
    "source_verification_status",
    "domain_expert_verified",
    "qa_review_status",
    "training_ready",
    "reviewer_decision",
    "reviewer_notes",
    "question_sha256",
    "answer_sha256",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ساخت QA Table منبع‌دار برای بخش 3-3 کتاب Odian "
            "از روی Knowledge Table بررسی‌شده."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="مسیر Knowledge Table با فرمت JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="پوشه خروجی QA Table",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"فایل ورودی پیدا نشد: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در خط {line_number}: {error}"
                ) from error
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("ي", "ی").replace("ك", "ک")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_formula(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def join_list(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value or "")


def add_formula(answer: str, formula: str) -> str:
    if not formula:
        return answer
    return f"{answer} رابطه مرجع: ${formula}$."


def build_direct_answer(record: dict[str, Any]) -> tuple[str, list[str]]:
    answer = normalize_text(record["statement_fa"])
    support = ["statement_fa"]

    formula = record.get("formula_latex", "")
    if formula:
        answer = add_formula(answer, formula)
        support.append("formula_latex")

    return answer, support


def build_reasoning_question(
    record: dict[str, Any],
    label: str,
) -> tuple[str, str, list[str]]:
    knowledge_type = record["knowledge_type"]
    formula = record.get("formula_latex", "")
    assumptions = normalize_text(record.get("assumptions_fa", ""))
    cause = normalize_text(record.get("cause_fa", ""))
    effect = normalize_text(record.get("effect_fa", ""))
    statement = normalize_text(record["statement_fa"])
    variables = record.get("variables_json", {})

    if knowledge_type in {
        "formula",
        "derived_formula",
        "typical_value",
        "effect_prediction",
    }:
        question = (
            f"رابطه یا وابستگی مربوط به «{label}» چه چیزی را نشان می‌دهد "
            "و تحت چه شرطی باید تفسیر شود؟"
        )
        parts = [statement]
        support = ["statement_fa"]
        if formula:
            parts.append(f"رابطه: ${formula}$.")
            support.append("formula_latex")
        if variables:
            variable_text = "؛ ".join(
                f"{key}: {value}"
                for key, value in variables.items()
            )
            parts.append(f"تعریف نمادها: {variable_text}.")
            support.append("variables_json")
        if assumptions:
            parts.append(f"فرض یا شرط اعتبار: {assumptions}.")
            support.append("assumptions_fa")
        if effect:
            parts.append(f"پیامد: {effect}.")
            support.append("effect_fa")
        return question, " ".join(parts), support

    if knowledge_type in {
        "measurement_method",
        "measurement_principle",
        "measurement_setup",
        "measurement_advantage",
        "instrument_capability",
    }:
        question = (
            f"اصل کار یا مزیت «{label}» چیست و برای معتبر بودن آن "
            "چه شرطی باید رعایت شود؟"
        )
        parts = [statement]
        support = ["statement_fa"]
        if assumptions:
            parts.append(f"شرط اعتبار: {assumptions}.")
            support.append("assumptions_fa")
        if effect:
            parts.append(f"خروجی یا مزیت: {effect}.")
            support.append("effect_fa")
        return question, " ".join(parts), support

    if knowledge_type in {"scope", "limitation", "measurement_limitation"}:
        question = (
            f"دامنه کاربرد یا محدودیت «{label}» چیست و دلیل آن چگونه "
            "توضیح داده می‌شود؟"
        )
        parts = [statement]
        support = ["statement_fa"]
        if cause:
            parts.append(f"دلیل: {cause}.")
            support.append("cause_fa")
        if effect:
            parts.append(f"نتیجه: {effect}.")
            support.append("effect_fa")
        if assumptions:
            parts.append(f"شرط: {assumptions}.")
            support.append("assumptions_fa")
        return question, " ".join(parts), support

    if knowledge_type == "assumption":
        question = (
            f"فرض «{label}» چه نقشی در تحلیل دارد و نباید با چه برداشتی "
            "اشتباه گرفته شود؟"
        )
        parts = [statement]
        support = ["statement_fa"]
        if assumptions:
            parts.append(f"دامنه فرض: {assumptions}.")
            support.append("assumptions_fa")
        if effect:
            parts.append(f"نتیجه استفاده از فرض: {effect}.")
            support.append("effect_fa")
        return question, " ".join(parts), support

    if knowledge_type in {"mechanism", "explanation"}:
        question = (
            f"علت و پیامد مکانیزمی «{label}» را بر اساس منبع توضیح دهید."
        )
        parts = [statement]
        support = ["statement_fa"]
        if cause:
            parts.append(f"علت: {cause}.")
            support.append("cause_fa")
        if effect:
            parts.append(f"پیامد: {effect}.")
            support.append("effect_fa")
        return question, " ".join(parts), support

    if knowledge_type == "definition":
        question = (
            f"«{label}» را تعریف کنید و مرز آن را با برداشت نادرست رایج "
            "مشخص کنید."
        )
        parts = [statement]
        support = ["statement_fa"]
        if effect:
            parts.append(f"نتیجه تعریف: {effect}.")
            support.append("effect_fa")
        return question, " ".join(parts), support

    question = (
        f"نکته علمی اصلی درباره «{label}» چیست و چه پیامدی دارد؟"
    )
    parts = [statement]
    support = ["statement_fa"]
    if cause:
        parts.append(f"علت: {cause}.")
        support.append("cause_fa")
    if effect:
        parts.append(f"پیامد: {effect}.")
        support.append("effect_fa")
    if assumptions:
        parts.append(f"شرط: {assumptions}.")
        support.append("assumptions_fa")
    return question, " ".join(parts), support


def build_misconception_qa(
    record: dict[str, Any],
    label: str,
) -> tuple[str, str, list[str]]:
    misconception = normalize_text(record.get("common_error_fa", ""))

    if not misconception:
        misconception = (
            f"مفهوم «{label}» را می‌توان بدون توجه به فرض‌ها و دامنه "
            "اعتبار آن به همه سامانه‌ها تعمیم داد"
        )

    question = (
        f"آیا این برداشت درست است: «{misconception}»؟ "
        "پاسخ را با استناد به مفهوم صحیح اصلاح کنید."
    )

    answer = (
        "خیر، این برداشت درست نیست. "
        + normalize_text(record["statement_fa"])
    )
    support = ["statement_fa", "common_error_fa"]

    formula = record.get("formula_latex", "")
    if formula:
        answer = add_formula(answer, formula)
        support.append("formula_latex")

    assumptions = normalize_text(record.get("assumptions_fa", ""))
    if assumptions:
        answer += f" شرط یا دامنه اعتبار: {assumptions}."
        support.append("assumptions_fa")

    return question, answer, support


def assign_split(record_id: str) -> str:
    if record_id in HOLDOUT_RECORD_IDS:
        return "holdout"
    if record_id in VALIDATION_RECORD_IDS:
        return "validation"
    return "train"


def build_qa_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda item: item["record_id"]):
        record_id = record["record_id"]
        label = TOPIC_LABELS[record_id]
        split = assign_split(record_id)

        direct_answer, direct_support = build_direct_answer(record)
        reasoning_question, reasoning_answer, reasoning_support = (
            build_reasoning_question(record, label)
        )

        canonical_formula = record.get("formula_latex", "")
        if (
            canonical_formula
            and canonical_formula not in reasoning_answer
        ):
            reasoning_answer = add_formula(
                reasoning_answer,
                canonical_formula,
            )
            if "formula_latex" not in reasoning_support:
                reasoning_support.append("formula_latex")

        misconception_question, misconception_answer, misconception_support = (
            build_misconception_qa(record, label)
        )

        variants = [
            {
                "variant": 1,
                "question_type": "direct_recall",
                "difficulty": "easy",
                "question_fa": DIRECT_QUESTIONS[record_id],
                "answer_fa": direct_answer,
                "support": direct_support,
            },
            {
                "variant": 2,
                "question_type": "reasoning_and_conditions",
                "difficulty": "medium",
                "question_fa": reasoning_question,
                "answer_fa": reasoning_answer,
                "support": reasoning_support,
            },
            {
                "variant": 3,
                "question_type": "misconception_correction",
                "difficulty": "medium",
                "question_fa": misconception_question,
                "answer_fa": misconception_answer,
                "support": misconception_support,
            },
        ]

        for item in variants:
            qa_id = (
                f"odian_3_3_qa_{int(record_id.rsplit('_', 1)[1]):03d}_"
                f"v{item['variant']}"
            )
            question = normalize_text(item["question_fa"])
            answer = normalize_text(item["answer_fa"])

            row = {
                "qa_id": qa_id,
                "split": split,
                "leakage_group": record_id,
                "source_record_id": record_id,
                "source_chunk_ids": record.get("source_chunk_ids", []),
                "printed_pages": record.get("printed_pages", []),
                "pdf_pages": record.get("pdf_pages", []),
                "subsection": record["subsection"],
                "subtopic": record["subtopic"],
                "knowledge_type": record["knowledge_type"],
                "question_type": item["question_type"],
                "difficulty": item["difficulty"],
                "question_fa": question,
                "answer_fa": answer,
                "canonical_formula_latex": record.get("formula_latex", ""),
                "assumptions_fa": record.get("assumptions_fa", ""),
                "evidence_statement_fa": record["statement_fa"],
                "answer_support_fields": item["support"],
                "answerable_from_source": True,
                "refusal_expected": False,
                "source_verification_status": record["verification_status"],
                "domain_expert_verified": bool(
                    record.get("domain_expert_verified", False)
                ),
                "qa_review_status": "pending_domain_expert_review",
                "training_ready": False,
                "reviewer_decision": "",
                "reviewer_notes": "",
                "question_sha256": sha256_text(question),
                "answer_sha256": sha256_text(answer),
            }
            rows.append(row)

    return rows


def validate_input_records(records: list[dict[str, Any]]) -> None:
    errors: list[str] = []

    if len(records) != EXPECTED_RECORD_COUNT:
        errors.append(
            f"تعداد رکوردهای ورودی باید {EXPECTED_RECORD_COUNT} باشد، "
            f"اما {len(records)} است."
        )

    ids = [record.get("record_id") for record in records]
    if len(ids) != len(set(ids)):
        errors.append("شناسه رکورد علمی تکراری وجود دارد.")

    missing_labels = set(ids) - set(TOPIC_LABELS)
    missing_questions = set(ids) - set(DIRECT_QUESTIONS)
    extra_labels = set(TOPIC_LABELS) - set(ids)
    extra_questions = set(DIRECT_QUESTIONS) - set(ids)

    if missing_labels:
        errors.append(f"برچسب موضوع برای این رکوردها موجود نیست: {sorted(missing_labels)}")
    if missing_questions:
        errors.append(f"سؤال مستقیم برای این رکوردها موجود نیست: {sorted(missing_questions)}")
    if extra_labels:
        errors.append(f"برچسب اضافه و بدون رکورد: {sorted(extra_labels)}")
    if extra_questions:
        errors.append(f"سؤال مستقیم اضافه و بدون رکورد: {sorted(extra_questions)}")

    if VALIDATION_RECORD_IDS & HOLDOUT_RECORD_IDS:
        errors.append("رکورد مشترک میان validation و holdout وجود دارد.")

    unknown_split_ids = (
        VALIDATION_RECORD_IDS | HOLDOUT_RECORD_IDS
    ) - set(ids)
    if unknown_split_ids:
        errors.append(
            f"رکوردهای تقسیم‌بندی‌شده در ورودی وجود ندارند: "
            f"{sorted(unknown_split_ids)}"
        )

    for record in records:
        record_id = record["record_id"]

        if record.get("verification_status") != (
            "source_reviewed_against_uploaded_text"
        ):
            errors.append(
                f"{record_id}: وضعیت منبع مورد انتظار نیست."
            )

        if bool(record.get("domain_expert_verified", False)):
            errors.append(
                f"{record_id}: نباید پیش از تأیید متخصص، verified باشد."
            )

        if not normalize_text(record.get("statement_fa", "")):
            errors.append(f"{record_id}: statement_fa خالی است.")

        if record_id in KNOWN_FORMULAS:
            actual = normalize_formula(record.get("formula_latex", ""))
            expected = normalize_formula(KNOWN_FORMULAS[record_id])
            if actual != expected:
                errors.append(
                    f"{record_id}: فرمول کلیدی با مقدار مورد انتظار "
                    "تطابق ندارد."
                )

    if errors:
        print("اعتبارسنجی ورودی ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)


def validate_qa_rows(
    records: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []

    if len(rows) != EXPECTED_QA_COUNT:
        errors.append(
            f"تعداد QAها باید {EXPECTED_QA_COUNT} باشد، اما {len(rows)} است."
        )

    qa_ids = [row["qa_id"] for row in rows]
    if len(qa_ids) != len(set(qa_ids)):
        errors.append("شناسه QA تکراری وجود دارد.")

    question_hashes = [row["question_sha256"] for row in rows]
    if len(question_hashes) != len(set(question_hashes)):
        errors.append("سؤال دقیقاً تکراری در جدول وجود دارد.")

    per_record = Counter(row["source_record_id"] for row in rows)
    for record in records:
        count = per_record[record["record_id"]]
        if count != QA_PER_RECORD:
            errors.append(
                f"{record['record_id']}: باید {QA_PER_RECORD} QA داشته باشد، "
                f"اما {count} دارد."
            )

    split_sets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_sets[row["split"]].add(row["source_record_id"])

        if not row["question_fa"].strip():
            errors.append(f"{row['qa_id']}: سؤال خالی است.")
        if not row["answer_fa"].strip():
            errors.append(f"{row['qa_id']}: پاسخ خالی است.")
        if len(row["question_fa"]) < 15:
            errors.append(f"{row['qa_id']}: سؤال بیش از حد کوتاه است.")
        if len(row["answer_fa"]) < 20:
            errors.append(f"{row['qa_id']}: پاسخ بیش از حد کوتاه است.")
        if row["answerable_from_source"] is not True:
            errors.append(f"{row['qa_id']}: باید answerable باشد.")
        if row["refusal_expected"] is not False:
            errors.append(f"{row['qa_id']}: نباید refusal باشد.")
        if row["training_ready"] is not False:
            errors.append(
                f"{row['qa_id']}: پیش از مرور متخصص نباید training_ready باشد."
            )
        if row["qa_review_status"] != "pending_domain_expert_review":
            errors.append(f"{row['qa_id']}: وضعیت مرور QA نامعتبر است.")

        answer_lower = row["answer_fa"].lower()
        for phrase in REFUSAL_PHRASES:
            if phrase in answer_lower:
                errors.append(
                    f"{row['qa_id']}: عبارت امتناع نامناسب در پاسخ وجود دارد."
                )

        formula = row["canonical_formula_latex"]
        if formula and formula not in row["answer_fa"]:
            warnings.append(
                f"{row['qa_id']}: فرمول canonical در متن پاسخ نیامده است."
            )

    if split_sets["train"] & split_sets["validation"]:
        errors.append("نشت رکورد میان train و validation وجود دارد.")
    if split_sets["train"] & split_sets["holdout"]:
        errors.append("نشت رکورد میان train و holdout وجود دارد.")
    if split_sets["validation"] & split_sets["holdout"]:
        errors.append("نشت رکورد میان validation و holdout وجود دارد.")

    expected_subsections = {
        "3-3a", "3-3b", "3-3c", "3-3c-1", "3-3c-2", "3-3c-3"
    }
    for split in ("train", "validation", "holdout"):
        present = {
            row["subsection"]
            for row in rows
            if row["split"] == split
        }
        missing = expected_subsections - present
        if missing:
            warnings.append(
                f"split={split}: زیربخش‌های بدون نمونه: {sorted(missing)}"
            )

    report = {
        "passed": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "expected_record_count": EXPECTED_RECORD_COUNT,
        "actual_record_count": len(records),
        "expected_qa_count": EXPECTED_QA_COUNT,
        "actual_qa_count": len(rows),
        "qa_per_record": QA_PER_RECORD,
        "split_record_counts": {
            split: len(record_ids)
            for split, record_ids in sorted(split_sets.items())
        },
        "split_qa_counts": dict(
            sorted(Counter(row["split"] for row in rows).items())
        ),
        "question_type_counts": dict(
            sorted(Counter(row["question_type"] for row in rows).items())
        ),
        "subsection_counts": dict(
            sorted(Counter(row["subsection"] for row in rows).items())
        ),
        "important_note": (
            "تمام QAها از Knowledge Table منبع‌دار ساخته شده‌اند. "
            "هیچ نمونه امتناع یا insufficient-context در این جدول وجود ندارد. "
            "همه نمونه‌های یک رکورد علمی در یک split نگه داشته شده‌اند تا "
            "نشت مفهومی مستقیم میان train، validation و holdout رخ ندهد. "
            "training_ready تا پیش از مرور متخصص دامنه false باقی می‌ماند."
        ),
    }

    if errors:
        print("اعتبارسنجی QA Table ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)

    return report


def json_ready_row(row: dict[str, Any]) -> dict[str, Any]:
    return row


def csv_ready_row(row: dict[str, Any]) -> dict[str, Any]:
    csv_row = dict(row)
    for field in (
        "source_chunk_ids",
        "printed_pages",
        "pdf_pages",
        "answer_support_fields",
    ):
        csv_row[field] = join_list(row[field])

    for field in (
        "answerable_from_source",
        "refusal_expected",
        "domain_expert_verified",
        "training_ready",
    ):
        csv_row[field] = str(row[field]).lower()

    return csv_row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(json_ready_row(row), ensure_ascii=False)
                + "\n"
            )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(csv_ready_row(row))


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)

    validate_input_records(records)
    rows = build_qa_rows(records)
    report = validate_qa_rows(records, rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_jsonl = args.output_dir / "qa_3_3_all_source_grounded.jsonl"
    all_csv = args.output_dir / "qa_3_3_all_source_grounded.csv"
    train_jsonl = args.output_dir / "qa_3_3_train.jsonl"
    validation_jsonl = args.output_dir / "qa_3_3_validation.jsonl"
    holdout_jsonl = args.output_dir / "qa_3_3_holdout.jsonl"
    summary_json = args.output_dir / "qa_3_3_summary.json"
    audit_json = args.output_dir / "qa_3_3_audit_report.json"

    write_jsonl(all_jsonl, rows)
    write_csv(all_csv, rows)

    split_paths = {
        "train": train_jsonl,
        "validation": validation_jsonl,
        "holdout": holdout_jsonl,
    }
    for split, path in split_paths.items():
        write_jsonl(
            path,
            [row for row in rows if row["split"] == split],
        )

    summary = {
        "book": records[0]["book"],
        "chapter": records[0]["chapter"],
        "section": records[0]["section"],
        "source_knowledge_record_count": len(records),
        "qa_count": len(rows),
        "qa_per_record": QA_PER_RECORD,
        "split_qa_counts": report["split_qa_counts"],
        "split_record_counts": report["split_record_counts"],
        "question_type_counts": report["question_type_counts"],
        "subsection_counts": report["subsection_counts"],
        "source_verification_status": (
            "source_reviewed_against_uploaded_text"
        ),
        "domain_expert_verified": False,
        "training_ready": False,
        "important_note": report["important_note"],
    }

    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    audit_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 76)
    print("QA Table بخش 3-3 با موفقیت ساخته و اعتبارسنجی شد.")
    print("All JSONL:", all_jsonl)
    print("All CSV:", all_csv)
    print("Train:", train_jsonl)
    print("Validation:", validation_jsonl)
    print("Holdout:", holdout_jsonl)
    print("Summary:", summary_json)
    print("Audit:", audit_json)
    print()
    print("تعداد رکوردهای علمی:", len(records))
    print("تعداد کل QAها:", len(rows))
    print("توزیع QA:", report["split_qa_counts"])
    print("توزیع رکوردهای منبع:", report["split_record_counts"])
    print("تعداد خطاهای اعتبارسنجی:", report["error_count"])
    print("تعداد هشدارها:", report["warning_count"])
    print()
    print("وضعیت:")
    print("- تمام QAها answerable_from_source=true هستند.")
    print("- هیچ نمونه refusal در این جدول وجود ندارد.")
    print("- نشت مستقیم source_record میان splitها وجود ندارد.")
    print("- training_ready تا مرور متخصص دامنه false است.")


if __name__ == "__main__":
    main()
