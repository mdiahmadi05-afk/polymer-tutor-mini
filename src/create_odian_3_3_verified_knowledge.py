from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(
    "data/knowledge/odian_ch3/review/knowledge_review_queue_3_3.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/knowledge/odian_ch3/verified"
)

FIELDS = [
    "record_id",
    "book_id",
    "book",
    "author",
    "edition",
    "chapter",
    "section",
    "subsection",
    "printed_pages",
    "pdf_pages",
    "source_chunk_ids",
    "source_review_ids",
    "source_anchor",
    "equation_ids",
    "topic",
    "subtopic",
    "knowledge_type",
    "statement_fa",
    "formula_latex",
    "variables_json",
    "assumptions_fa",
    "cause_fa",
    "effect_fa",
    "common_error_fa",
    "keywords_fa",
    "keywords_en",
    "evidence_mode",
    "verification_status",
    "domain_expert_verified",
    "formula_review_status",
    "review_notes",
]

COMMON = {
    "book_id": "odian_4e",
    "book": "Principles of Polymerization",
    "author": "George Odian",
    "edition": "Fourth Edition",
    "chapter": "3",
    "section": "3-3",
    "topic": "radical_chain_polymerization",
    "evidence_mode": "direct",
    "verification_status": "source_reviewed_against_uploaded_text",
    "domain_expert_verified": False,
}


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
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSON نامعتبر در خط {line_number}: {exc}"
                ) from exc
    return rows


def record(
    number: int,
    *,
    subsection: str,
    chunks: list[str],
    subtopic: str,
    knowledge_type: str,
    statement: str,
    formula: str = "",
    variables: dict[str, str] | None = None,
    assumptions: str = "",
    cause: str = "",
    effect: str = "",
    common_error: str = "",
    keywords_fa: str = "",
    keywords_en: str = "",
    equations: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        **COMMON,
        "record_id": f"odian_3_3_k_{number:03d}",
        "subsection": subsection,
        "source_chunk_ids": chunks,
        "source_anchor": ", ".join(equations or chunks),
        "equation_ids": equations or [],
        "subtopic": subtopic,
        "knowledge_type": knowledge_type,
        "statement_fa": statement,
        "formula_latex": formula,
        "variables_json": variables or {},
        "assumptions_fa": assumptions,
        "cause_fa": cause,
        "effect_fa": effect,
        "common_error_fa": common_error,
        "keywords_fa": keywords_fa,
        "keywords_en": keywords_en,
        "formula_review_status": (
            "manual_latex_reconstruction_from_equation_anchor"
            if formula
            else "not_applicable"
        ),
        "review_notes": notes,
    }


RECORDS = [
    record(
        1,
        subsection="3-3a",
        chunks=["odian_3_3_p204_c01"],
        subtopic="sequence_of_events",
        knowledge_type="mechanism",
        statement=(
            "پلیمریزاسیون زنجیره‌ای رادیکالی یک واکنش زنجیره‌ای با سه "
            "مرحله اصلی آغازش، رشد و اختتام است."
        ),
        effect="این سه مرحله چارچوب مکانیزمی تحلیل سرعت و رشد زنجیر را تشکیل می‌دهند.",
        common_error="انتقال زنجیر را یکی از سه مرحله بنیادی این توالی در نظر گرفتن.",
        keywords_fa="آغازش، رشد، اختتام، واکنش زنجیره‌ای",
        keywords_en="initiation, propagation, termination, chain reaction",
    ),
    record(
        2,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c01"],
        subtopic="initiator_dissociation",
        knowledge_type="formula",
        statement=(
            "بخش نخست آغازش معمولاً شکافت همولیتیکی گونه آغازگر I و تولید "
            "یک جفت رادیکال اولیه R• است."
        ),
        formula=r"I \xrightarrow{k_d} 2R^{\bullet}",
        variables={
            "I": "گونه آغازگر",
            "R•": "رادیکال اولیه یا رادیکال آغازگر",
            "kd": "ثابت سرعت تفکیک آغازگر",
        },
        cause="شکافت همولیتیکی پیوند در مولکول آغازگر.",
        effect="دو رادیکال اولیه برای ورود به مرحله بعدی آغازش تولید می‌شوند.",
        common_error="آغازگر را کاتالیزوری دانستن که در واکنش مصرف نمی‌شود.",
        keywords_fa="تفکیک همولیتیکی، آغازگر، رادیکال اولیه",
        keywords_en="homolytic dissociation, initiator, primary radical",
        equations=["3-13"],
        notes="نقطه رادیکالی در متن OCR حذف شده بود و در LaTeX بازسازی شد.",
    ),
    record(
        3,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c01"],
        subtopic="chain_initiation",
        knowledge_type="formula",
        statement=(
            "بخش دوم آغازش، افزوده‌شدن رادیکال اولیه به نخستین مولکول مونومر "
            "و تشکیل رادیکال آغازکننده زنجیر M1• است."
        ),
        formula=r"R^{\bullet}+M \xrightarrow{k_i} M_1^{\bullet}",
        variables={
            "R•": "رادیکال اولیه",
            "M": "مولکول مونومر",
            "M1•": "رادیکال آغازکننده زنجیر",
            "ki": "ثابت سرعت مرحله آغازش",
        },
        assumptions="رادیکال اولیه بتواند به پیوند قابل پلیمریزاسیون مونومر اضافه شود.",
        effect="مرکز فعال زنجیری برای شروع رشد تشکیل می‌شود.",
        common_error="تولید رادیکال اولیه و افزودن آن به مونومر را یک رویداد واحد بدون دو مرحله دانستن.",
        keywords_fa="رادیکال آغازکننده زنجیر، نخستین مونومر، ثابت آغازش",
        keywords_en="chain-initiating radical, first monomer, initiation constant",
        equations=["3-14a", "3-14b"],
        notes="ساختار اختصاصی CH2=CHY در منبع آمده است؛ رکورد حاضر شکل عمومی را نگه می‌دارد.",
    ),
    record(
        4,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c02"],
        subtopic="radical_terminology",
        knowledge_type="definition",
        statement=(
            "R• رادیکال آغازگر یا رادیکال اولیه نامیده می‌شود و از گونه "
            "آغازکننده زنجیر M1• متمایز است."
        ),
        common_error="R• و M1• را یک گونه رادیکالی با نقش یکسان در نظر گرفتن.",
        keywords_fa="رادیکال اولیه، رادیکال آغازگر، رادیکال زنجیری",
        keywords_en="primary radical, initiator radical, chain radical",
    ),
    record(
        5,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c02"],
        subtopic="propagation_reaction",
        knowledge_type="formula",
        statement=(
            "در رشد، رادیکال زنجیری با افزودن پیاپی تعداد زیادی مولکول مونومر "
            "به زنجیر بلندتر تبدیل می‌شود."
        ),
        formula=r"M_n^{\bullet}+M \xrightarrow{k_p} M_{n+1}^{\bullet}",
        variables={
            "Mn•": "رادیکال زنجیری با n واحد مونومری",
            "M": "مونومر",
            "Mn+1•": "رادیکال زنجیری پس از افزودن یک مونومر",
            "kp": "ثابت سرعت رشد",
        },
        effect="در هر گام یک واحد مونومری به طول زنجیر افزوده می‌شود.",
        common_error="رشد را واکنش میان دو زنجیر مرده دانستن.",
        keywords_fa="رشد زنجیر، افزودن پیاپی مونومر، ثابت رشد",
        keywords_en="chain propagation, successive monomer addition, propagation constant",
        equations=["3-15a", "3-15b", "3-15c", "3-15d"],
        notes="نقطه رادیکالی در LaTeX بازسازی شد.",
    ),
    record(
        6,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c02"],
        subtopic="propagating_radical_identity",
        knowledge_type="mechanism",
        statement=(
            "هر گام رشد رادیکال جدیدی ایجاد می‌کند که از نظر نوع مرکز فعال "
            "همانند رادیکال پیشین است، اما یک واحد مونومری بزرگ‌تر است."
        ),
        cause="مرکز رادیکالی پس از افزودن مونومر در انتهای زنجیر باقی می‌ماند.",
        effect="زنجیر می‌تواند گام‌های رشد بسیار زیادی را به‌صورت متوالی طی کند.",
        common_error="تصور اینکه مرکز فعال در هر گام رشد مصرف و بدون جانشین حذف می‌شود.",
        keywords_fa="حفظ مرکز فعال، افزایش یک واحد مونومری",
        keywords_en="active-center retention, one-monomer growth",
    ),
    record(
        7,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c03"],
        subtopic="propagation_rate_constant",
        knowledge_type="typical_value",
        statement=(
            "رشد زنجیر تا ابعاد پلیمر پرجرم سریع است و مقدار kp برای بسیاری از "
            "مونومرها در محدوده تقریبی 10^2 تا 10^4 L mol^-1 s^-1 قرار دارد."
        ),
        formula=r"k_p\sim10^2\text{--}10^4\ \mathrm{L\,mol^{-1}\,s^{-1}}",
        variables={"kp": "ثابت سرعت رشد"},
        assumptions="این بازه یک مقدار معمول برای بسیاری از مونومرهاست و مقدار همگانی ثابت نیست.",
        effect="تعداد زیادی گام رشد می‌تواند در زمان کوتاه انجام شود.",
        common_error="بازه معمول kp را برای همه مونومرها و همه شرایط مقدار ثابت دانستن.",
        keywords_fa="بازه ثابت رشد، رشد سریع، پلیمر پرجرم",
        keywords_en="propagation-constant range, rapid propagation, high polymer",
        equations=["3-15"],
    ),
    record(
        8,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c03", "odian_3_3_p206_c01"],
        subtopic="bimolecular_termination",
        knowledge_type="mechanism",
        statement=(
            "اختتام با واکنش دومولکولی میان دو رادیکال زنجیری و نابودی مراکز "
            "رادیکالی رخ می‌دهد."
        ),
        cause="برخورد دو رادیکال زنجیری.",
        effect="رشد زنجیرهای درگیر متوقف می‌شود.",
        common_error="اختتام را واکنش تک‌مولکولی یک رادیکال بدون شریک دانستن.",
        keywords_fa="اختتام دومولکولی، نابودی رادیکال، توقف رشد",
        keywords_en="bimolecular termination, radical annihilation, growth cessation",
        equations=["3-16", "3-17", "3-18"],
    ),
    record(
        9,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c03", "odian_3_3_p206_c01"],
        subtopic="termination_by_coupling",
        knowledge_type="formula",
        statement=(
            "در اختتام از نوع ترکیب یا جفت‌شدن، دو رادیکال زنجیری به یک "
            "مولکول پلیمر مرده با طول مجموع دو زنجیر تبدیل می‌شوند."
        ),
        formula=r"M_n^{\bullet}+M_m^{\bullet}\xrightarrow{k_{tc}}M_{n+m}",
        variables={
            "Mn•, Mm•": "دو رادیکال زنجیری",
            "M(n+m)": "محصول مرده حاصل از جفت‌شدن",
            "ktc": "ثابت سرعت اختتام از طریق ترکیب",
        },
        effect="یک مولکول پلیمر مرده از دو زنجیر رادیکالی ساخته می‌شود.",
        common_error="در اختتام ترکیبی دو مولکول مرده مستقل در نظر گرفتن.",
        keywords_fa="اختتام ترکیبی، جفت‌شدن، یک محصول مرده",
        keywords_en="termination by combination, coupling, single dead product",
        equations=["3-16a", "3-17a"],
        notes="نقطه‌های رادیکالی در متن OCR بازسازی شدند.",
    ),
    record(
        10,
        subsection="3-3a",
        chunks=["odian_3_3_p205_c03", "odian_3_3_p206_c01"],
        subtopic="termination_by_disproportionation",
        knowledge_type="mechanism",
        statement=(
            "در نامتناسب‌شدن، انتقال هیدروژن بتا میان دو رادیکال زنجیری "
            "دو مولکول پلیمر مرده ایجاد می‌کند که یکی اشباع و دیگری غیراشباع است."
        ),
        formula=r"M_n^{\bullet}+M_m^{\bullet}\xrightarrow{k_{td}}M_n+M_m",
        variables={
            "Mn•, Mm•": "دو رادیکال زنجیری",
            "ktd": "ثابت سرعت اختتام از طریق نامتناسب‌شدن",
        },
        cause="انتقال اتم هیدروژن بتا از یک زنجیر به مرکز رادیکالی زنجیر دیگر.",
        effect="دو مولکول پلیمر مرده تشکیل می‌شود.",
        common_error="نامتناسب‌شدن را مانند ترکیب، تشکیل یک زنجیر با طول مجموع دانستن.",
        keywords_fa="نامتناسب‌شدن، انتقال هیدروژن بتا، محصول اشباع و غیراشباع",
        keywords_en="disproportionation, beta-hydrogen transfer, saturated and unsaturated products",
        equations=["3-16b", "3-17b"],
        notes="رابطه عمومی 3-17b نوع محصولات را نشان می‌دهد؛ جزئیات اشباع/غیراشباع از متن همان Chunk آمده است.",
    ),
    record(
        11,
        subsection="3-3a",
        chunks=["odian_3_3_p206_c01"],
        subtopic="generic_termination",
        knowledge_type="formula",
        statement=(
            "وقتی مسیر دقیق اختتام مشخص نشود، می‌توان واکنش دو رادیکال زنجیری "
            "را به‌صورت کلی به تولید پلیمر مرده نمایش داد."
        ),
        formula=r"M_n^{\bullet}+M_m^{\bullet}\xrightarrow{k_t}\text{dead polymer}",
        variables={
            "kt": "ثابت سرعت کلی اختتام",
            "dead polymer": "پلیمری که رشد رادیکالی آن متوقف شده است",
        },
        assumptions="تفکیک سهم ترکیب و نامتناسب‌شدن برای تحلیل موردنظر لازم نباشد.",
        common_error="رابطه کلی را به‌معنای تشکیل الزاماً یک مولکول محصول تعبیر کردن.",
        keywords_fa="اختتام کلی، پلیمر مرده، ثابت کلی اختتام",
        keywords_en="generic termination, dead polymer, overall termination constant",
        equations=["3-18"],
    ),
    record(
        12,
        subsection="3-3a",
        chunks=["odian_3_3_p206_c01"],
        subtopic="overall_termination_constant",
        knowledge_type="formula",
        statement=(
            "ثابت سرعت کلی اختتام میانگین سهم‌دار ثابت‌های اختتام از طریق "
            "ترکیب و نامتناسب‌شدن است."
        ),
        formula=r"k_t=\alpha k_{tc}+(1-\alpha)k_{td}",
        variables={
            "kt": "ثابت سرعت کلی اختتام",
            "ktc": "ثابت سرعت اختتام ترکیبی",
            "ktd": "ثابت سرعت اختتام نامتناسب‌شدن",
            "α": "کسر اختتام از طریق ترکیب",
            "1-α": "کسر اختتام از طریق نامتناسب‌شدن",
        },
        assumptions="دو مسیر اصلی اختتام همان ترکیب و نامتناسب‌شدن باشند.",
        common_error="kt را جمع ساده ktc و ktd بدون درنظرگرفتن کسر هر مسیر نوشتن.",
        keywords_fa="ثابت کلی اختتام، کسر ترکیب، کسر نامتناسب‌شدن",
        keywords_en="overall termination constant, coupling fraction, disproportionation fraction",
        equations=["3-19"],
    ),
    record(
        13,
        subsection="3-3a",
        chunks=["odian_3_3_p206_c02"],
        subtopic="dead_polymer",
        knowledge_type="definition",
        statement=(
            "پلیمر مرده به زنجیری گفته می‌شود که رشد رادیکالی آن متوقف شده است."
        ),
        effect="زنجیر دیگر از مسیر رشد رادیکالی واحد مونومری دریافت نمی‌کند.",
        common_error="پلیمر مرده را الزاماً پلیمر تخریب‌شده یا تجزیه‌شده دانستن.",
        keywords_fa="پلیمر مرده، توقف رشد، حذف مرکز فعال",
        keywords_en="dead polymer, cessation of growth, loss of active center",
    ),
    record(
        14,
        subsection="3-3a",
        chunks=["odian_3_3_p206_c02", "odian_3_3_p208_c01"],
        subtopic="propagation_vs_termination",
        knowledge_type="explanation",
        statement=(
            "با وجود اینکه kt معمولاً چند مرتبه بزرگ‌تر از kp است، رشد ادامه "
            "می‌یابد؛ زیرا غلظت رادیکال‌ها بسیار کم است و سرعت پلیمریزاسیون "
            "تنها با توان منفی یک‌دوم به kt وابسته است."
        ),
        formula=r"R_p\propto k_t^{-1/2}",
        variables={"Rp": "سرعت پلیمریزاسیون", "kt": "ثابت سرعت اختتام"},
        cause="اختتام به برخورد دو رادیکال کم‌غلظت نیاز دارد.",
        effect="بزرگ‌بودن عددی kt به‌تنهایی مانع تشکیل زنجیرهای بلند نمی‌شود.",
        common_error="فقط از مقایسه kt و kp نتیجه گرفتن که اختتام باید همیشه بر رشد غالب باشد.",
        keywords_fa="رقابت رشد و اختتام، غلظت کم رادیکال، وابستگی ریشه دوم",
        keywords_en="propagation-termination competition, low radical concentration, square-root dependence",
        equations=["3-25"],
    ),
    record(
        15,
        subsection="3-3b",
        chunks=["odian_3_3_p206_c03"],
        subtopic="detailed_mechanism_scope",
        knowledge_type="scope",
        statement=(
            "روابط 3-13 تا 3-19 سازوکار تفصیلی پلیمریزاسیون زنجیره‌ای "
            "آغازشده با رادیکال آزاد را تشکیل می‌دهند."
        ),
        effect="این مجموعه روابط مبنای استخراج رابطه سینتیکی سرعت پلیمریزاسیون است.",
        keywords_fa="سازوکار تفصیلی، روابط 3-13 تا 3-19",
        keywords_en="detailed mechanism, equations 3-13 through 3-19",
        equations=["3-13", "3-14", "3-15", "3-16", "3-17", "3-18", "3-19"],
    ),
    record(
        16,
        subsection="3-3b",
        chunks=["odian_3_3_p206_c03"],
        subtopic="equal_reactivity_assumption",
        knowledge_type="assumption",
        statement=(
            "برای استخراج رابطه سرعت، فرض می‌شود kp و kt به اندازه رادیکال "
            "زنجیری وابسته نیستند؛ این همان فرض واکنش‌پذیری برابر است."
        ),
        assumptions="رادیکال‌های زنجیری پس از اندازه‌های بسیار کوچک رفتار سینتیکی تقریباً مشابه داشته باشند.",
        effect="می‌توان همه گام‌های رشد را با یک kp و همه اختتام‌ها را با یک kt نمایش داد.",
        common_error="فرض واکنش‌پذیری برابر را به معنی برابر بودن kp و kt تعبیر کردن.",
        keywords_fa="فرض واکنش‌پذیری برابر، استقلال از طول زنجیر",
        keywords_en="equal-reactivity assumption, chain-length independence",
        equations=["3-15", "3-16", "3-17", "3-18"],
    ),
    record(
        17,
        subsection="3-3b",
        chunks=["odian_3_3_p206_c03"],
        subtopic="small_radical_size_effect",
        knowledge_type="scope",
        statement=(
            "رادیکال‌های بسیار کوچک از رادیکال‌های در حال رشد واکنش‌پذیرترند، "
            "اما اثر اندازه در حدود دایمر یا تریمر از بین می‌رود."
        ),
        effect="ناهمسانی رادیکال‌های بسیار کوچک معمولاً اثر مهمی بر سینتیک کلی رشد ندارد.",
        common_error="فرض واکنش‌پذیری برابر را از نخستین رادیکال بدون هیچ استثنایی دقیق دانستن.",
        keywords_fa="اثر اندازه رادیکال، دایمر، تریمر",
        keywords_en="radical-size effect, dimer, trimer",
    ),
    record(
        18,
        subsection="3-3b",
        chunks=["odian_3_3_p206_c04"],
        subtopic="equal_reactivity_limitations",
        knowledge_type="limitation",
        statement=(
            "فرض واکنش‌پذیری برابر نسبت به اندازه رادیکال‌های در حال رشد "
            "محدودیت‌هایی دارد و یک تقریب مطلق و بدون استثنا نیست."
        ),
        common_error="استفاده از فرض واکنش‌پذیری برابر به‌عنوان قانون دقیق در همه تبدیل‌ها و شرایط.",
        keywords_fa="محدودیت فرض واکنش‌پذیری برابر",
        keywords_en="limitations of equal-reactivity assumption",
        notes="جزئیات محدودیت‌ها در بخش‌های بعدی کتاب ارجاع شده و در متن حاضر تشریح نشده‌اند.",
    ),
    record(
        19,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c01"],
        subtopic="monomer_disappearance_rate",
        knowledge_type="formula",
        statement=(
            "مونومر هم در آغازش و هم در رشد مصرف می‌شود؛ بنابراین سرعت کل "
            "ناپدیدشدن مونومر برابر مجموع سرعت‌های آغازش و رشد است."
        ),
        formula=r"-\frac{d[M]}{dt}=R_i+R_p",
        variables={
            "[M]": "غلظت مونومر",
            "Ri": "سرعت مصرف مونومر در آغازش",
            "Rp": "سرعت مصرف مونومر در رشد",
        },
        common_error="علامت منفی مصرف مونومر را حذف‌کردن یا Ri و Rp را یک کمیت دانستن.",
        keywords_fa="سرعت ناپدیدشدن مونومر، سرعت آغازش، سرعت رشد",
        keywords_en="monomer disappearance rate, initiation rate, propagation rate",
        equations=["3-20"],
        notes="علامت منفی در OCR افتاده بود و از معنای ناپدیدشدن مونومر بازسازی شد.",
    ),
    record(
        20,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c01"],
        subtopic="high_polymer_rate_approximation",
        knowledge_type="approximation",
        statement=(
            "در فرایندی که پلیمر پرجرم تولید می‌کند، تعداد مونومرهای مصرفی در "
            "آغازش نسبت به رشد بسیار کم است؛ پس سرعت پلیمریزاسیون با تقریب خوب "
            "برابر سرعت رشد است."
        ),
        formula=r"-\frac{d[M]}{dt}\approx R_p",
        variables={"[M]": "غلظت مونومر", "Rp": "سرعت رشد"},
        assumptions="مصرف مونومر در آغازش نسبت به تعداد بسیار زیاد گام‌های رشد ناچیز باشد.",
        effect="Ri از رابطه عملی سرعت پلیمریزاسیون حذف می‌شود.",
        common_error="این تقریب را برای واکنش‌های بدون رشد زنجیری طولانی نیز به‌کاربردن.",
        keywords_fa="تقریب پلیمر پرجرم، سرعت پلیمریزاسیون، سرعت رشد",
        keywords_en="high-polymer approximation, polymerization rate, propagation rate",
        equations=["3-21"],
    ),
    record(
        21,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c01"],
        subtopic="propagation_rate_expression",
        knowledge_type="formula",
        statement=(
            "سرعت رشد برابر حاصل‌ضرب ثابت سرعت رشد، غلظت مونومر و غلظت کل "
            "رادیکال‌های زنجیری است."
        ),
        formula=r"R_p=k_p[M][M^{\bullet}]",
        variables={
            "Rp": "سرعت رشد یا پلیمریزاسیون",
            "kp": "ثابت سرعت رشد",
            "[M]": "غلظت مونومر",
            "[M•]": "غلظت کل رادیکال‌های زنجیری",
        },
        assumptions="ثابت رشد برای رادیکال‌های زنجیری مختلف یکسان فرض شود.",
        effect="افزایش [M] یا [M•] در شرایط ثابت Rp را افزایش می‌دهد.",
        common_error="[M] و [M•] را یک غلظت دانستن.",
        keywords_fa="رابطه سرعت رشد، غلظت مونومر، غلظت رادیکال",
        keywords_en="propagation-rate expression, monomer concentration, radical concentration",
        equations=["3-22"],
        notes="نقطه رادیکالی در LaTeX بازسازی شد.",
    ),
    record(
        22,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c02"],
        subtopic="total_chain_radical_concentration",
        knowledge_type="definition",
        statement=(
            "[M•] غلظت کل همه رادیکال‌های زنجیری، از M1• و زنجیرهای بلندتر، است."
        ),
        common_error="[M•] را فقط غلظت رادیکال M1• یا فقط یک طول زنجیر دانستن.",
        keywords_fa="غلظت کل رادیکال‌های زنجیری، M1 و بزرگ‌تر",
        keywords_en="total chain-radical concentration, M1 and larger radicals",
        equations=["3-22"],
    ),
    record(
        23,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c02"],
        subtopic="radical_concentration_measurement",
        knowledge_type="measurement_limitation",
        statement=(
            "غلظت رادیکال‌ها بسیار کم، در حدود 10^-8 mol L^-1، است و اندازه‌گیری "
            "کمی مستقیم آن دشوار است."
        ),
        formula=r"[M^{\bullet}]\sim10^{-8}\ \mathrm{mol\,L^{-1}}",
        variables={"[M•]": "غلظت کل رادیکال‌های زنجیری"},
        cause="رادیکال‌ها واسطه‌های بسیار واکنش‌پذیر و کم‌غلظت‌اند.",
        effect="برای حذف [M•] از رابطه سرعت از فرض حالت پایا استفاده می‌شود.",
        common_error="غلظت رادیکال را هم‌مرتبه با غلظت مونومر فرض‌کردن.",
        keywords_fa="غلظت بسیار کم رادیکال، دشواری اندازه‌گیری",
        keywords_en="very low radical concentration, measurement difficulty",
        equations=["3-22"],
    ),
    record(
        24,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c02"],
        subtopic="steady_state_assumption",
        knowledge_type="assumption",
        statement=(
            "در فرض حالت پایا، غلظت رادیکال‌ها پس از افزایش اولیه تقریباً ثابت "
            "می‌شود و نرخ تغییر آن نزدیک صفر است؛ در نتیجه Ri و Rt برابرند."
        ),
        formula=r"\frac{d[M^{\bullet}]}{dt}\approx0,\qquad R_i=R_t",
        variables={
            "[M•]": "غلظت کل رادیکال‌ها",
            "Ri": "سرعت ایجاد رادیکال‌ها",
            "Rt": "سرعت نابودی رادیکال‌ها",
        },
        assumptions="دوره گذر اولیه سپری شده و شرایط واکنش در بازه تحلیل تغییر شدید نکند.",
        cause="سرعت تولید و نابودی رادیکال‌ها پس از گذر کوتاه متعادل می‌شود.",
        effect="غلظت رادیکال را می‌توان بر حسب Ri و kt نوشت.",
        common_error="حالت پایا را به معنی صفر بودن غلظت رادیکال یا توقف واکنش دانستن.",
        keywords_fa="فرض حالت پایا، برابری آغازش و اختتام",
        keywords_en="steady-state assumption, equality of initiation and termination",
        equations=["3-23"],
    ),
    record(
        25,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c02", "odian_3_3_p207_c03"],
        subtopic="termination_rate_expression",
        knowledge_type="formula",
        statement=(
            "در اختتام دومولکولی، هر رویداد دو رادیکال را حذف می‌کند؛ بنابراین "
            "سرعت نابودی رادیکال‌ها برابر 2kt[M•]^2 است."
        ),
        formula=r"R_t=2k_t[M^{\bullet}]^2",
        variables={
            "Rt": "سرعت نابودی رادیکال‌ها",
            "kt": "ثابت سرعت اختتام",
            "[M•]": "غلظت کل رادیکال‌های زنجیری",
        },
        assumptions="اختتام از نظر سینتیکی دومولکولی باشد.",
        cause="در هر برخورد اختتام دو رادیکال از بین می‌روند.",
        effect="Rt با مربع غلظت رادیکال متناسب است.",
        common_error="ضریب 2 را بدون توجه به قرارداد تعریف Rt حذف‌کردن.",
        keywords_fa="سرعت اختتام، ضریب دو، وابستگی درجه دوم",
        keywords_en="termination rate, factor of two, second-order dependence",
        equations=["3-23"],
    ),
    record(
        26,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c03"],
        subtopic="steady_state_timescale",
        knowledge_type="timescale",
        statement=(
            "پلیمریزاسیون‌های معمول می‌توانند پس از دوره‌ای کوتاه، که حداکثر "
            "در حدود یک دقیقه گزارش شده است، به حالت پایا برسند."
        ),
        assumptions="این زمان یک توصیف معمول است و برای همه سامانه‌ها مقدار دقیق یکسان نیست.",
        effect="در بخش عمده زمان واکنش می‌توان از تقریب حالت پایا استفاده کرد.",
        common_error="فرض حالت پایا را از لحظه صفر و بدون دوره گذر معتبر دانستن.",
        keywords_fa="زمان رسیدن به حالت پایا، دوره گذر",
        keywords_en="time to steady state, transient period",
    ),
    record(
        27,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c03"],
        subtopic="termination_mode_kinetics",
        knowledge_type="scope",
        statement=(
            "برای رابطه سرعت اختتام لازم نیست مشخص شود مسیر اختتام ترکیب است یا "
            "نامتناسب‌شدن، زیرا هر دو از یک فرم سینتیکی دومولکولی پیروی می‌کنند."
        ),
        assumptions="هر دو مسیر با برخورد دو رادیکال زنجیری رخ دهند.",
        effect="رابطه 2kt[M•]^2 برای سرعت کلی نابودی رادیکال‌ها قابل استفاده است.",
        common_error="تفاوت محصول‌های دو مسیر را الزاماً به تفاوت مرتبه سینتیکی آن‌ها تعمیم دادن.",
        keywords_fa="سینتیک مشترک اختتام، ترکیب، نامتناسب‌شدن",
        keywords_en="common termination kinetics, combination, disproportionation",
        equations=["3-23"],
    ),
    record(
        28,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c03", "odian_3_3_p207_c04"],
        subtopic="factor_two_convention",
        knowledge_type="convention",
        statement=(
            "به‌کاربردن ضریب 2 برای واکنش‌هایی که رادیکال‌ها را به‌صورت جفت "
            "ایجاد یا نابود می‌کنند، قرارداد ترجیحی IUPAC است؛ بااین‌حال در "
            "برخی منابع قدیمی این قرارداد رعایت نشده است."
        ),
        effect="مقایسه روابط سینتیکی منابع مختلف نیازمند توجه به قرارداد تعریف سرعت است.",
        common_error="اختلاف ضریب 2 میان دو منبع را بدون بررسی قرارداد، تناقض علمی تلقی کردن.",
        keywords_fa="قرارداد IUPAC، ضریب دو، ادبیات قدیمی",
        keywords_en="IUPAC convention, factor of two, older literature",
        equations=["3-13", "3-23"],
    ),
    record(
        29,
        subsection="3-3b",
        chunks=["odian_3_3_p207_c04"],
        subtopic="steady_state_radical_concentration",
        knowledge_type="derived_formula",
        statement=(
            "با حل رابطه حالت پایا، غلظت رادیکال‌های زنجیری برابر ریشه دوم "
            "نسبت Ri به 2kt به دست می‌آید."
        ),
        formula=r"[M^{\bullet}]=\left(\frac{R_i}{2k_t}\right)^{1/2}",
        variables={
            "[M•]": "غلظت کل رادیکال‌های زنجیری",
            "Ri": "سرعت ایجاد رادیکال‌ها",
            "kt": "ثابت سرعت اختتام",
        },
        assumptions="حالت پایا و اختتام دومولکولی برقرار باشد و قرارداد ضریب 2 به‌کار رود.",
        effect="[M•] با Ri به توان یک‌دوم و با kt به توان منفی یک‌دوم تغییر می‌کند.",
        common_error="وابستگی خطی [M•] به Ri یا فراموش‌کردن ضریب 2.",
        keywords_fa="غلظت حالت پایای رادیکال، وابستگی ریشه دوم",
        keywords_en="steady-state radical concentration, square-root dependence",
        equations=["3-24"],
        notes="کسر و توان در OCR شکسته بود و از Eq. 3-23 بازسازی شد.",
    ),
    record(
        30,
        subsection="3-3b",
        chunks=["odian_3_3_p208_c01"],
        subtopic="polymerization_rate_equation",
        knowledge_type="derived_formula",
        statement=(
            "جایگذاری غلظت حالت پایای رادیکال‌ها در رابطه رشد، رابطه نهایی سرعت "
            "پلیمریزاسیون را بر حسب [M]، Ri، kp و kt می‌دهد."
        ),
        formula=r"R_p=k_p[M]\left(\frac{R_i}{2k_t}\right)^{1/2}",
        variables={
            "Rp": "سرعت پلیمریزاسیون",
            "kp": "ثابت سرعت رشد",
            "[M]": "غلظت مونومر",
            "Ri": "سرعت آغازش مؤثر",
            "kt": "ثابت سرعت اختتام",
        },
        assumptions="حالت پایا، اختتام دومولکولی، واکنش‌پذیری برابر و ناچیزبودن مصرف مونومر در آغازش برقرار باشند.",
        effect="Rp با [M] و kp مستقیم، با Ri ریشه دوم و با kt معکوس ریشه دوم است.",
        common_error="Rp را با Ri خطی یا با kt معکوس خطی دانستن.",
        keywords_fa="رابطه نهایی سرعت پلیمریزاسیون، حالت پایا",
        keywords_en="final polymerization-rate equation, steady state",
        equations=["3-25"],
        notes="ساختار کسر و توان از شماره رابطه و متن پیرامون بازسازی شد.",
    ),
    record(
        31,
        subsection="3-3b",
        chunks=["odian_3_3_p208_c01"],
        subtopic="square_root_initiation_dependence",
        knowledge_type="effect_prediction",
        statement=(
            "سرعت پلیمریزاسیون با ریشه دوم سرعت آغازش تغییر می‌کند؛ بنابراین "
            "دو برابرکردن Ri، Rp را فقط به اندازه sqrt(2) افزایش می‌دهد."
        ),
        formula=r"\frac{R_{p,2}}{R_{p,1}}=\left(\frac{R_{i,2}}{R_{i,1}}\right)^{1/2}=\sqrt{2}",
        variables={
            "Rp,1, Rp,2": "سرعت‌های پلیمریزاسیون پیش و پس از تغییر",
            "Ri,1, Ri,2": "سرعت‌های آغازش پیش و پس از تغییر",
        },
        assumptions="[M]، kp، kt و سایر شرایط ثابت بمانند.",
        cause="اختتام دومولکولی با افزایش غلظت رادیکال‌ها سریع‌تر می‌شود.",
        effect="افزایش Ri بازده خطی در Rp ایجاد نمی‌کند.",
        common_error="دو برابرشدن Ri را مساوی دو برابرشدن Rp دانستن.",
        keywords_fa="وابستگی ریشه دوم، دو برابرشدن سرعت آغازش",
        keywords_en="square-root dependence, doubling initiation rate",
        equations=["3-25"],
    ),
    record(
        32,
        subsection="3-3c",
        chunks=["odian_3_3_p208_c02"],
        subtopic="experimental_rate_principle",
        knowledge_type="measurement_principle",
        statement=(
            "سرعت پلیمریزاسیون را می‌توان با اندازه‌گیری تغییر خاصیتی دنبال کرد "
            "که میان مونومر و پلیمر متفاوت است؛ مانند حلالیت، چگالی، ضریب شکست "
            "یا جذب طیفی."
        ),
        assumptions="خاصیت انتخاب‌شده با تبدیل به‌صورت قابل‌اندازه‌گیری تغییر کند.",
        effect="تبدیل بر حسب زمان و در نتیجه Rp به‌طور تجربی تعیین می‌شود.",
        common_error="هر خاصیت فیزیکی را بدون کالیبراسیون و بدون تفاوت کافی میان مونومر و پلیمر مناسب دانستن.",
        keywords_fa="تعیین تجربی سرعت، حلالیت، چگالی، ضریب شکست، جذب طیفی",
        keywords_en="experimental rate determination, solubility, density, refractive index, spectral absorption",
    ),
    record(
        33,
        subsection="3-3c",
        chunks=["odian_3_3_p208_c02"],
        subtopic="method_selection",
        knowledge_type="method_selection",
        statement=(
            "مناسب‌بودن هر روش اندازه‌گیری به نوع پلیمریزاسیون و دقت و صحت آن "
            "در تبدیل‌های کم، متوسط و زیاد بستگی دارد."
        ),
        effect="یک روش ممکن است فقط در بخشی از بازه تبدیل یا برای یک سازوکار پلیمریزاسیون مناسب باشد.",
        common_error="یک روش آزمایشگاهی را برای همه سامانه‌ها و همه درصدهای تبدیل به یک اندازه معتبر دانستن.",
        keywords_fa="انتخاب روش، دقت، صحت، بازه تبدیل",
        keywords_en="method selection, precision, accuracy, conversion range",
    ),
    record(
        34,
        subsection="3-3c",
        chunks=["odian_3_3_p208_c03"],
        subtopic="continuous_monitoring",
        knowledge_type="measurement_advantage",
        statement=(
            "برخی روش‌ها بدون متوقف‌کردن واکنش، تبدیل را در طول زمان روی همان "
            "نمونه دنبال می‌کنند."
        ),
        effect="نیاز به برداشت و خاتمه‌دادن نمونه‌های جداگانه کاهش می‌یابد.",
        common_error="پایش پیوسته را با اندازه‌گیری فقط ابتدا و انتهای واکنش یکسان دانستن.",
        keywords_fa="پایش پیوسته، بدون توقف واکنش، همان نمونه",
        keywords_en="continuous monitoring, no reaction stopping, same sample",
    ),
    record(
        35,
        subsection="3-3c-1",
        chunks=["odian_3_3_p208_c04"],
        subtopic="polymer_isolation_method",
        knowledge_type="measurement_method",
        statement=(
            "در روش جداسازی فیزیکی، در زمان‌های مختلف از سامانه نمونه‌برداری می‌شود، "
            "پلیمر معمولاً با افزودن غیرحلال رسوب داده، خشک و وزن می‌شود."
        ),
        assumptions="پلیمر و مونومر تفاوت حلالیت کافی برای جداسازی داشته باشند.",
        effect="جرم پلیمر در برابر زمان برای تعیین پیشرفت واکنش به دست می‌آید.",
        common_error="وزن‌کردن رسوب مرطوب یا ناخالص را معادل جرم پلیمر خشک دانستن.",
        keywords_fa="رسوب‌دهی با غیرحلال، خشک‌کردن، وزن‌کردن پلیمر",
        keywords_en="nonsolvent precipitation, drying, polymer weighing",
    ),
    record(
        36,
        subsection="3-3c-1",
        chunks=["odian_3_3_p208_c04"],
        subtopic="isolation_method_scope",
        knowledge_type="scope",
        statement=(
            "روش جداسازی و وزن‌کردن عمدتاً برای پلیمریزاسیون زنجیره‌ای مناسب است، "
            "زیرا سامانه شامل مونومر و پلیمر پرجرم با تفاوت زیاد حلالیت است؛ این "
            "روش معمولاً برای پلیمریزاسیون مرحله‌ای مناسب نیست."
        ),
        cause="در مرحله‌ای تا تبدیل‌های بسیار بالا مجموعه‌ای از مونومرها و محصولات کم‌جرم با تفاوت حلالیت محدود وجود دارد.",
        effect="جداسازی انتخابی پلیمر مرحله‌ای در تبدیل‌های پایین و متوسط دشوار می‌شود.",
        common_error="روش رسوب‌دهی پلیمر پرجرم را بدون تغییر برای الیگومرهای مرحله‌ای به‌کاربردن.",
        keywords_fa="دامنه روش جداسازی، زنجیره‌ای، مرحله‌ای، تفاوت حلالیت",
        keywords_en="isolation-method scope, chain polymerization, step polymerization, solubility difference",
    ),
    record(
        37,
        subsection="3-3c-1",
        chunks=["odian_3_3_p208_c04"],
        subtopic="isolation_method_limitations",
        knowledge_type="limitation",
        statement=(
            "روش جداسازی، خشک‌کردن و وزن‌کردن زمان‌بر است و برای نتیجه دقیق به "
            "مراقبت زیاد نیاز دارد."
        ),
        effect="خطا در رسوب‌دهی، شست‌وشو، خشک‌کردن یا وزن‌کردن می‌تواند نتیجه تبدیل را منحرف کند.",
        common_error="این روش را ذاتاً دقیق و مستقل از کیفیت عملیات آزمایشگاهی فرض‌کردن.",
        keywords_fa="روش زمان‌بر، دقت عملیاتی، خطای وزن‌کردن",
        keywords_en="time-consuming method, operational care, weighing error",
    ),
    record(
        38,
        subsection="3-3c-1",
        chunks=["odian_3_3_p208_c04", "odian_3_3_p208_c05"],
        subtopic="byproduct_monitoring",
        knowledge_type="measurement_method",
        statement=(
            "برای بسیاری از پلیمریزاسیون‌های مرحله‌ای می‌توان محصول جانبی کوچک را "
            "به‌طور پیوسته پایش کرد؛ برای نمونه در پلی‌استری‌شدن دی‌ال و دی‌اسید بالای "
            "100°C، آب تقطیرشده در تله مدرج جمع‌آوری و حجم آن اندازه‌گیری می‌شود."
        ),
        assumptions="محصول جانبی فرار به‌طور قابل جمع‌آوری و استوکیومتری تولید شود.",
        effect="پیشرفت واکنش مرحله‌ای از مقدار محصول جانبی بر حسب زمان محاسبه می‌شود.",
        common_error="از دست‌رفتن آب یا وجود آب اولیه را در محاسبه تبدیل نادیده گرفتن.",
        keywords_fa="پایش محصول جانبی، پلی‌استری‌شدن، آب تقطیرشده، تله مدرج",
        keywords_en="byproduct monitoring, polyesterification, distilled water, calibrated trap",
    ),
    record(
        39,
        subsection="3-3c-2",
        chunks=["odian_3_3_p208_c06"],
        subtopic="functional_group_analysis",
        knowledge_type="measurement_method",
        statement=(
            "در پلیمریزاسیون مرحله‌ای، تحلیل شیمیایی غلظت گروه‌های عاملی واکنش‌نداده "
            "در زمان‌های مختلف می‌تواند سرعت واکنش را تعیین کند؛ نمونه‌ها شامل تیترکردن "
            "گروه کربوکسیل با باز استاندارد و تحلیل گروه هیدروکسیل با انیدرید استیک‌اند."
        ),
        assumptions="گروه عاملی انتخاب‌شده به‌طور انتخابی و کمی قابل اندازه‌گیری باشد.",
        effect="کاهش غلظت گروه‌های عاملی به تبدیل و سرعت پلیمریزاسیون مرتبط می‌شود.",
        common_error="تغییر غلظت ناشی از نمونه‌برداری یا واکنش‌های جانبی را به‌طور کامل به پلیمریزاسیون نسبت دادن.",
        keywords_fa="تحلیل گروه عاملی، تیتر کربوکسیل، انیدرید استیک، گروه هیدروکسیل",
        keywords_en="functional-group analysis, carboxyl titration, acetic anhydride, hydroxyl group",
    ),
    record(
        40,
        subsection="3-3c-2",
        chunks=["odian_3_3_p208_c06"],
        subtopic="bromine_titration",
        knowledge_type="measurement_method",
        statement=(
            "سرعت پلیمریزاسیون زنجیره‌ای مونومرهای وینیلی را می‌توان با تیترکردن "
            "پیوندهای دوگانه واکنش‌نداده با برم دنبال کرد."
        ),
        assumptions="برم به‌طور کمی با پیوند دوگانه باقیمانده واکنش دهد و تداخل جانبی کنترل شود.",
        effect="کاهش پیوند دوگانه مونومر در برابر زمان اندازه‌گیری می‌شود.",
        common_error="مصرف برم در واکنش‌های جانبی را به‌طور کامل به تبدیل مونومر نسبت دادن.",
        keywords_fa="تیتر برم، پیوند دوگانه، مونومر وینیلی",
        keywords_en="bromine titration, double bond, vinyl monomer",
    ),
    record(
        41,
        subsection="3-3c-2",
        chunks=["odian_3_3_p209_c01"],
        subtopic="spectroscopic_monitoring",
        knowledge_type="measurement_method",
        statement=(
            "ناپدیدشدن مونومر یا پیدایش پلیمر را می‌توان با طیف‌سنجی IR، UV، NMR "
            "و روش‌های طیفی دیگر از کاهش سیگنال مونومر یا افزایش سیگنال پلیمر دنبال کرد."
        ),
        assumptions="سیگنال‌های مناسب به مونومر یا پلیمر نسبت داده و به‌طور کمی تحلیل شوند.",
        effect="تبدیل بر حسب زمان از شدت سیگنال‌های طیفی تعیین می‌شود.",
        common_error="هر تغییر شدت سیگنال را بدون تصحیح خط پایه و هم‌پوشانی به تبدیل نسبت دادن.",
        keywords_fa="پایش طیفی، IR، UV، NMR، سیگنال مونومر و پلیمر",
        keywords_en="spectroscopic monitoring, IR, UV, NMR, monomer and polymer signals",
    ),
    record(
        42,
        subsection="3-3c-2",
        chunks=["odian_3_3_p209_c01"],
        subtopic="styrene_nmr_example",
        knowledge_type="example",
        statement=(
            "در مثال پلیمریزاسیون استایرن، سیگنال‌های پروتونی مونومر در 5.23، "
            "5.73 و 6.71 ppm با تبدیل کاهش می‌یابند و سیگنال‌های CH2 و CH پلیمر "
            "در 1.44 و 1.84 ppm ظاهر می‌شوند."
        ),
        assumptions="اختصاص سیگنال‌ها صحیح و شرایط NMR برای مقایسه کمی کنترل شده باشد.",
        effect="کاهش مونومر و تشکیل پلیمر به‌صورت هم‌زمان قابل مشاهده است.",
        common_error="مقادیر جابه‌جایی شیمیایی این مثال را برای همه مونومرهای وینیلی تعمیم دادن.",
        keywords_fa="NMR استایرن، جابه‌جایی شیمیایی، سیگنال مونومر و پلیمر",
        keywords_en="styrene NMR, chemical shift, monomer and polymer signals",
    ),
    record(
        43,
        subsection="3-3c-2",
        chunks=["odian_3_3_p209_c01"],
        subtopic="spectroscopy_accuracy_and_continuity",
        knowledge_type="measurement_advantage",
        statement=(
            "دقت روش طیفی هنگامی بالا است که سیگنال‌های مونومر و پلیمر هم‌پوشانی "
            "نداشته باشند؛ طیف‌سنجی همچنین می‌تواند واکنش را بدون برداشت دوره‌ای نمونه "
            "به‌طور پیوسته پایش کند."
        ),
        assumptions="سیگنال‌ها تفکیک‌پذیر و پاسخ دستگاه پایدار باشد.",
        effect="خطای جداسازی سیگنال کاهش و پایش درجا ممکن می‌شود.",
        common_error="وجود هم‌پوشانی شدید سیگنال‌ها را بدون مدل تفکیک کمی نادیده گرفتن.",
        keywords_fa="هم‌پوشانی سیگنال، دقت طیف‌سنجی، پایش درجا",
        keywords_en="signal overlap, spectroscopy accuracy, in-situ monitoring",
    ),
    record(
        44,
        subsection="3-3c-2",
        chunks=["odian_3_3_p209_c02"],
        subtopic="in_situ_spectrometer_setup",
        knowledge_type="measurement_setup",
        statement=(
            "در پایش درجا، پلیمریزاسیون می‌تواند در لوله نمونه داخل طیف‌سنج و در "
            "دمای واکنش موردنظر انجام شود و تحلیل‌های دوره‌ای تبدیل را بر حسب زمان بدهند."
        ),
        assumptions="نمونه در دمای واکنش به تعادل برسد و ابزار با شرایط واکنش سازگار باشد.",
        effect="واکنش بدون انتقال نمونه میان راکتور و دستگاه دنبال می‌شود.",
        common_error="دمای واقعی نمونه داخل طیف‌سنج را با دمای تنظیم‌شده بدون کالیبراسیون برابر فرض‌کردن.",
        keywords_fa="لوله نمونه داخل طیف‌سنج، دمای واکنش، تحلیل دوره‌ای",
        keywords_en="sample tube in spectrometer, reaction temperature, periodic spectral analysis",
    ),
    record(
        45,
        subsection="3-3c-2",
        chunks=["odian_3_3_p209_c02"],
        subtopic="simultaneous_conversion_molecular_weight",
        knowledge_type="instrument_capability",
        statement=(
            "ابزارهای پیشرفته‌تر می‌توانند تبدیل و وزن مولکولی را به‌طور هم‌زمان در "
            "طول زمان واکنش تحلیل کنند."
        ),
        effect="رابطه میان پیشرفت واکنش و تکامل وزن مولکولی مستقیماً بررسی می‌شود.",
        common_error="اندازه‌گیری هم‌زمان را به معنی استفاده از هر طیف‌سنج معمولی بدون سامانه تحلیل تکمیلی دانستن.",
        keywords_fa="تحلیل هم‌زمان تبدیل و وزن مولکولی",
        keywords_en="simultaneous conversion and molecular-weight analysis",
    ),
    record(
        46,
        subsection="3-3c-3",
        chunks=["odian_3_3_p209_c03"],
        subtopic="dilatometry",
        knowledge_type="measurement_method",
        statement=(
            "دیلاتومتری از تغییر حجم هنگام پلیمریزاسیون استفاده می‌کند؛ برای برخی "
            "پلیمریزاسیون‌های زنجیره‌ای که تبدیل مونومر به پلیمر با جمع‌شدگی حجمی "
            "قابل‌توجه همراه است، حجم در ظرف مدرج بر حسب زمان ثبت می‌شود."
        ),
        assumptions="تغییر حجم به‌طور قابل‌توجه و کالیبره‌شده با تبدیل مرتبط باشد.",
        effect="تبدیل و سرعت واکنش از منحنی حجم-زمان تعیین می‌شود.",
        common_error="اثر دما، انبساط دستگاه یا حباب را به‌عنوان تغییر حجم ناشی از پلیمریزاسیون ثبت‌کردن.",
        keywords_fa="دیلاتومتری، جمع‌شدگی حجمی، ظرف مدرج، حجم-زمان",
        keywords_en="dilatometry, volume shrinkage, calibrated vessel, volume-time curve",
        notes="عبارت عددی مثال PMMA به‌دلیل ناسازگاری ظاهری جهت چگالی با جمع‌شدگی حجمی وارد رکورد نشد و نیازمند بررسی تصویری/ویرایشی است.",
    ),
    record(
        47,
        subsection="3-3c-3",
        chunks=["odian_3_3_p209_c03"],
        subtopic="dilatometry_limitation",
        knowledge_type="limitation",
        statement=(
            "دیلاتومتری برای پلیمریزاسیون مرحله‌ای معمول با محصول جانبی کوچک "
            "مناسب نیست، زیرا تغییر حجم خالص پلیمریزاسیون معمولاً قابل‌توجه نیست."
        ),
        cause="تشکیل و حضور محصول جانبی کوچک اثر حجمی فرایند را برای این روش نامناسب می‌کند.",
        common_error="دیلاتومتری را بدون بررسی موازنه حجم برای همه پلیمریزاسیون‌های مرحله‌ای به‌کاربردن.",
        keywords_fa="محدودیت دیلاتومتری، پلیمریزاسیون مرحله‌ای، محصول جانبی کوچک",
        keywords_en="dilatometry limitation, step polymerization, small-molecule byproduct",
    ),
    record(
        48,
        subsection="3-3c-3",
        chunks=["odian_3_3_p209_c03"],
        subtopic="dsc_conversion",
        knowledge_type="measurement_method",
        statement=(
            "گرمای پلیمریزاسیون را می‌توان با DSC به‌دقت اندازه گرفت و این گرما "
            "به‌طور مستقیم با تبدیل مرتبط است."
        ),
        assumptions="آنتالپی ویژه پلیمریزاسیون شناخته شده و واکنش‌های گرمازای جانبی کنترل شوند.",
        effect="تبدیل از گرمای آزادشده یا باقیمانده محاسبه می‌شود.",
        common_error="هر قله گرمازا را بدون تفکیک واکنش‌های جانبی به پلیمریزاسیون نسبت دادن.",
        keywords_fa="DSC، گرمای پلیمریزاسیون، تبدیل",
        keywords_en="DSC, heat of polymerization, conversion",
    ),
    record(
        49,
        subsection="3-3c-3",
        chunks=["odian_3_3_p209_c03"],
        subtopic="other_measurement_techniques",
        knowledge_type="measurement_method",
        statement=(
            "پراکندگی نور و اندازه‌گیری ضریب شکست نیز از روش‌های به‌کاررفته برای "
            "دنبال‌کردن پلیمریزاسیون هستند."
        ),
        assumptions="پاسخ اندازه‌گیری‌شده با پیشرفت واکنش همبستگی کالیبره‌شده داشته باشد.",
        common_error="ضریب شکست یا شدت پراکندگی را بدون تصحیح دما و ترکیب مستقیماً تبدیل دانستن.",
        keywords_fa="پراکندگی نور، ضریب شکست، پایش پلیمریزاسیون",
        keywords_en="light scattering, refractive index, polymerization monitoring",
    ),
]


def enrich_sources(records: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> None:
    by_chunk = {row["chunk_id"]: row for row in review_rows}

    for item in records:
        missing = [cid for cid in item["source_chunk_ids"] if cid not in by_chunk]
        if missing:
            raise ValueError(
                f"{item['record_id']}: Chunkهای منبع پیدا نشدند: {missing}"
            )

        source_rows = [by_chunk[cid] for cid in item["source_chunk_ids"]]
        item["source_review_ids"] = [row["review_id"] for row in source_rows]
        item["printed_pages"] = sorted({int(row["printed_page"]) for row in source_rows})
        item["pdf_pages"] = sorted({int(row["pdf_page_1_based"]) for row in source_rows})


def validate(records: list[dict[str, Any]], review_rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []

    if len(review_rows) != 21:
        errors.append(f"تعداد ردیف‌های Review باید ۲۱ باشد، اما {len(review_rows)} است.")

    if len(records) != 49:
        errors.append(f"تعداد رکوردهای Knowledge باید ۴۹ باشد، اما {len(records)} است.")

    ids = [item["record_id"] for item in records]
    expected_ids = {f"odian_3_3_k_{number:03d}" for number in range(1, 50)}

    if set(ids) != expected_ids:
        errors.append("شناسه‌ها باید از odian_3_3_k_001 تا odian_3_3_k_049 باشند.")
    if len(ids) != len(set(ids)):
        errors.append("شناسه Knowledge تکراری وجود دارد.")

    review_chunk_ids = {row["chunk_id"] for row in review_rows}
    used_chunk_ids: set[str] = set()

    for item in records:
        used_chunk_ids.update(item["source_chunk_ids"])

        missing_fields = [field for field in FIELDS if field not in item]
        if missing_fields:
            errors.append(f"{item['record_id']}: فیلدهای مفقود: {missing_fields}")

        if not item["statement_fa"].strip():
            errors.append(f"{item['record_id']}: statement_fa خالی است.")

        if not item["source_chunk_ids"]:
            errors.append(f"{item['record_id']}: منبع Chunk ندارد.")

        if not set(item["source_chunk_ids"]).issubset(review_chunk_ids):
            errors.append(f"{item['record_id']}: Chunk نامعتبر دارد.")

        if item["formula_latex"] and item["formula_review_status"] == "not_applicable":
            errors.append(f"{item['record_id']}: وضعیت فرمول نامعتبر است.")

        if item["domain_expert_verified"] is not False:
            errors.append(
                f"{item['record_id']}: domain_expert_verified باید تا بررسی متخصص False بماند."
            )

        if item["verification_status"] != "source_reviewed_against_uploaded_text":
            errors.append(f"{item['record_id']}: وضعیت بررسی منبع نامعتبر است.")

        if not all(204 <= page <= 209 for page in item["printed_pages"]):
            errors.append(f"{item['record_id']}: صفحه چاپی خارج از ۲۰۴ تا ۲۰۹ است.")

    if used_chunk_ids != review_chunk_ids:
        errors.append(
            "پوشش Chunkها کامل نیست. "
            f"استفاده‌نشده: {sorted(review_chunk_ids-used_chunk_ids)}"
        )

    required_equations = {
        "3-13", "3-14a", "3-15a", "3-16a", "3-16b", "3-17a", "3-17b",
        "3-18", "3-19", "3-20", "3-21", "3-22", "3-23", "3-24", "3-25",
    }
    found_equations = {eq for item in records for eq in item["equation_ids"]}
    missing_equations = required_equations - found_equations
    if missing_equations:
        errors.append(f"روابط ضروری پوشش داده نشده‌اند: {sorted(missing_equations)}")

    if errors:
        print("اعتبارسنجی ناموفق بود:")
        for error in errors:
            print("-", error)
        raise SystemExit(1)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for item in records:
            row = dict(item)
            for field in [
                "printed_pages", "pdf_pages", "source_chunk_ids", "source_review_ids",
                "equation_ids",
            ]:
                row[field] = ";".join(str(value) for value in row[field])
            row["variables_json"] = json.dumps(row["variables_json"], ensure_ascii=False)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    review_rows = load_jsonl(args.input)
    records = json.loads(json.dumps(RECORDS, ensure_ascii=False))

    enrich_sources(records, review_rows)
    validate(records, review_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "knowledge_3_3_source_reviewed.jsonl"
    csv_path = args.output_dir / "knowledge_3_3_source_reviewed.csv"
    summary_path = args.output_dir / "knowledge_3_3_summary.json"

    write_jsonl(jsonl_path, records)
    write_csv(csv_path, records)

    summary = {
        "record_count": len(records),
        "source_chunk_count": len({cid for item in records for cid in item["source_chunk_ids"]}),
        "printed_pages": sorted({page for item in records for page in item["printed_pages"]}),
        "subsection_counts": dict(Counter(item["subsection"] for item in records)),
        "knowledge_type_counts": dict(Counter(item["knowledge_type"] for item in records)),
        "formula_record_count": sum(bool(item["formula_latex"]) for item in records),
        "verification_status": "source_reviewed_against_uploaded_text",
        "domain_expert_verified": False,
        "important_note": (
            "رکوردها با متن آپلودشده تطبیق داده شده‌اند، اما تأیید نهایی متخصص دامنه "
            "هنوز انجام نشده است. فرمول‌ها از شماره روابط و متن پیرامون به LaTeX بازسازی شده‌اند."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 72)
    print("Knowledge Table بخش 3-3 ساخته شد.")
    print("JSONL:", jsonl_path)
    print("CSV:", csv_path)
    print("Summary:", summary_path)
    print("تعداد رکوردهای علمی:", len(records))
    print("تعداد Chunkهای منبع پوشش‌داده‌شده:", summary["source_chunk_count"])
    print("تعداد رکوردهای دارای فرمول:", summary["formula_record_count"])
    print("صفحات چاپی:", summary["printed_pages"])
    print("وضعیت: source_reviewed_against_uploaded_text")
    print("تأیید متخصص دامنه: انجام نشده")


if __name__ == "__main__":
    main()
