import csv
import json
from pathlib import Path


OUTPUT_DIR = Path("data/knowledge/odian_ch3")

CSV_PATH = OUTPUT_DIR / "knowledge_3_3b_rate_expression.csv"
JSONL_PATH = OUTPUT_DIR / "knowledge_3_3b_rate_expression.jsonl"


FIELDNAMES = [
    "record_id",
    "book_id",
    "book",
    "author",
    "edition",
    "chapter",
    "section",
    "printed_page_start",
    "printed_page_end",
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
    "source_anchor",
    "verification_status",
    "review_notes",
]


COMMON = {
    "book_id": "odian_4e",
    "book": "Principles of Polymerization",
    "author": "George Odian",
    "edition": "Fourth Edition",
    "chapter": "3",
    "section": "3-3b",
    "printed_page_start": 206,
    "printed_page_end": 207,
    "topic": "radical_chain_polymerization",
    "subtopic": "rate_expression",
    "evidence_mode": "direct",
    "source_anchor": "Chapter 3, Section 3-3b, Eqs. 3-21 to 3-25",
    "verification_status": "source_reviewed",
    "review_notes": "",
}


RECORDS = [
    {
        **COMMON,
        "record_id": "odian_3_3b_001",
        "knowledge_type": "definition",
        "statement_fa": (
            "در پلیمریزاسیون زنجیره‌ای رادیکالی، تعداد مولکول‌های "
            "مونومر مصرف‌شده در مرحله آغازش در مقایسه با مرحله رشد "
            "بسیار کم است؛ بنابراین سرعت پلیمریزاسیون با تقریب بسیار "
            "خوب برابر سرعت رشد در نظر گرفته می‌شود."
        ),
        "formula_latex": r"R_p=-\frac{d[M]}{dt}",
        "variables_json": json.dumps(
            {
                "Rp": "سرعت پلیمریزاسیون یا سرعت رشد",
                "[M]": "غلظت مونومر",
                "t": "زمان",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": (
            "پلیمریزاسیون به تولید پلیمر با جرم مولکولی زیاد منجر شود "
            "و مصرف مونومر در آغازش نسبت به رشد ناچیز باشد."
        ),
        "cause_fa": "بیشتر مصرف مونومر در واکنش‌های متوالی رشد رخ می‌دهد.",
        "effect_fa": "برای محاسبه سرعت کل، سهم آغازش از مصرف مونومر حذف می‌شود.",
        "common_error_fa": (
            "سرعت پلیمریزاسیون را مجموع مساوی سرعت آغازش و رشد در نظر گرفتن."
        ),
        "keywords_fa": "سرعت پلیمریزاسیون، سرعت رشد، مصرف مونومر",
        "keywords_en": "polymerization rate, propagation rate, monomer consumption",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_002",
        "knowledge_type": "formula",
        "statement_fa": (
            "سرعت رشد برابر حاصل‌ضرب ثابت سرعت رشد، غلظت مونومر "
            "و مجموع غلظت تمام رادیکال‌های زنجیری است."
        ),
        "formula_latex": r"R_p=k_p[M][M^\bullet]",
        "variables_json": json.dumps(
            {
                "Rp": "سرعت رشد یا پلیمریزاسیون",
                "kp": "ثابت سرعت رشد",
                "[M]": "غلظت مونومر",
                "[M•]": "غلظت کل رادیکال‌های زنجیری",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": (
            "ثابت سرعت رشد برای رادیکال‌های زنجیری با طول‌های مختلف "
            "یکسان در نظر گرفته شود."
        ),
        "cause_fa": "واکنش رشد میان مونومر و یک انتهای زنجیری رادیکالی انجام می‌شود.",
        "effect_fa": "افزایش [M] یا [M•] در شرایط ثابت موجب افزایش Rp می‌شود.",
        "common_error_fa": (
            "اشتباه‌گرفتن [M] مونومر با [M•] غلظت رادیکال‌های زنجیری."
        ),
        "keywords_fa": "ثابت رشد، غلظت مونومر، غلظت رادیکال",
        "keywords_en": "propagation constant, monomer concentration, radical concentration",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_003",
        "knowledge_type": "measurement_limitation",
        "statement_fa": (
            "رابطه مستقیم سرعت رشد به‌آسانی قابل استفاده نیست، زیرا "
            "غلظت رادیکال‌های زنجیری بسیار کم و اندازه‌گیری کمی آن دشوار است."
        ),
        "formula_latex": r"[M^\bullet]\sim10^{-8}\ \mathrm{mol\,L^{-1}}",
        "variables_json": json.dumps(
            {
                "[M•]": "غلظت کل رادیکال‌های زنجیری",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": "مرتبه غلظت ذکرشده برای پلیمریزاسیون‌های رادیکالی معمولی است.",
        "cause_fa": "رادیکال‌ها واسطه‌های بسیار فعال و کوتاه‌عمر هستند.",
        "effect_fa": (
            "برای حذف [M•] از رابطه سرعت، از فرض حالت پایا استفاده می‌شود."
        ),
        "common_error_fa": (
            "فرض‌کردن اینکه غلظت رادیکال‌ها به اندازه غلظت مونومر است."
        ),
        "keywords_fa": "غلظت کم رادیکال، اندازه‌گیری رادیکال، واسطه فعال",
        "keywords_en": "low radical concentration, radical measurement, reactive intermediate",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_004",
        "knowledge_type": "assumption",
        "statement_fa": (
            "در فرض حالت پایا، غلظت رادیکال‌ها پس از دوره اولیه کوتاه "
            "تقریباً ثابت می‌شود؛ بنابراین نرخ تشکیل رادیکال با نرخ "
            "ازبین‌رفتن آن برابر است."
        ),
        "formula_latex": r"\frac{d[M^\bullet]}{dt}\approx0,\qquad R_i=R_t",
        "variables_json": json.dumps(
            {
                "[M•]": "غلظت رادیکال‌های زنجیری",
                "Ri": "سرعت آغازش مؤثر رادیکال‌ها",
                "Rt": "سرعت اختتام رادیکال‌ها",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": (
            "سیستم از دوره آغازین گذشته باشد و شرایط پلیمریزاسیون "
            "در بازه موردنظر تغییر شدید نکند."
        ),
        "cause_fa": (
            "رادیکال‌ها با سرعت زیاد تشکیل و مصرف می‌شوند و غلظت آن‌ها "
            "در مقدار بسیار کوچکی پایدار می‌شود."
        ),
        "effect_fa": "امکان محاسبه [M•] از روی Ri و kt فراهم می‌شود.",
        "common_error_fa": (
            "حالت پایا را به معنی صفر بودن غلظت رادیکال یا توقف واکنش دانستن."
        ),
        "keywords_fa": "فرض حالت پایا، سرعت آغازش، سرعت اختتام",
        "keywords_en": "steady-state assumption, initiation rate, termination rate",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_005",
        "knowledge_type": "formula",
        "statement_fa": (
            "در اختتام دومولکولی، دو رادیکال در هر رویداد اختتام از بین "
            "می‌روند؛ بنابراین سرعت ازبین‌رفتن رادیکال‌ها دارای ضریب 2 است."
        ),
        "formula_latex": r"R_t=2k_t[M^\bullet]^2",
        "variables_json": json.dumps(
            {
                "Rt": "سرعت ازبین‌رفتن رادیکال‌ها",
                "kt": "ثابت سرعت اختتام",
                "[M•]": "غلظت کل رادیکال‌های زنجیری",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": "اختتام از نوع دومولکولی میان دو رادیکال زنجیری باشد.",
        "cause_fa": "هر واکنش اختتام هم‌زمان دو رادیکال را مصرف می‌کند.",
        "effect_fa": "سرعت اختتام با مربع غلظت رادیکال متناسب است.",
        "common_error_fa": (
            "حذف ضریب 2 بدون مشخص‌کردن قرارداد مورد استفاده برای تعریف Rt."
        ),
        "keywords_fa": "اختتام دومولکولی، ضریب دو، ثابت اختتام",
        "keywords_en": "bimolecular termination, factor of two, termination constant",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_006",
        "knowledge_type": "scope",
        "statement_fa": (
            "رابطه سینتیکی اختتام برای اختتام از طریق ترکیب و "
            "نامتناسب‌شدن یکسان است؛ زیرا در هر دو حالت برخورد دو "
            "رادیکال زنجیری موجب حذف آن‌ها می‌شود."
        ),
        "formula_latex": r"R_t=2k_t[M^\bullet]^2",
        "variables_json": json.dumps(
            {
                "Rt": "سرعت اختتام",
                "kt": "ثابت سرعت اختتام",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": (
            "هر دو مسیر اختتام از نظر سینتیکی دومولکولی باشند."
        ),
        "cause_fa": "هر دو مسیر به برخورد دو رادیکال نیاز دارند.",
        "effect_fa": (
            "برای استخراج رابطه سرعت نیازی به تعیین سهم ترکیب و "
            "نامتناسب‌شدن نیست."
        ),
        "common_error_fa": (
            "تصور اینکه نوع محصول اختتام الزاماً مرتبه سینتیکی را تغییر می‌دهد."
        ),
        "keywords_fa": "ترکیب، نامتناسب‌شدن، سینتیک اختتام",
        "keywords_en": "combination, disproportionation, termination kinetics",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_007",
        "knowledge_type": "derived_formula",
        "statement_fa": (
            "با برابر قراردادن سرعت آغازش و اختتام در حالت پایا، "
            "غلظت رادیکال‌های زنجیری از رابطه ریشه دوم Ri بر 2kt به دست می‌آید."
        ),
        "formula_latex": r"[M^\bullet]=\left(\frac{R_i}{2k_t}\right)^{1/2}",
        "variables_json": json.dumps(
            {
                "[M•]": "غلظت رادیکال‌های زنجیری",
                "Ri": "سرعت آغازش مؤثر",
                "kt": "ثابت سرعت اختتام",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": "فرض حالت پایا و اختتام دومولکولی برقرار باشد.",
        "cause_fa": r"از ترکیب روابط R_i=R_t و R_t=2k_t[M^\bullet]^2 حاصل می‌شود.",
        "effect_fa": (
            "غلظت رادیکال با ریشه دوم سرعت آغازش افزایش و با ریشه دوم "
            "ثابت اختتام کاهش می‌یابد."
        ),
        "common_error_fa": (
            "نوشتن وابستگی خطی [M•] به Ri یا فراموش‌کردن توان یک‌دوم."
        ),
        "keywords_fa": "غلظت رادیکال، ریشه دوم، حالت پایا",
        "keywords_en": "radical concentration, square-root dependence, steady state",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_008",
        "knowledge_type": "derived_formula",
        "statement_fa": (
            "جایگذاری غلظت حالت پایای رادیکال‌ها در رابطه رشد، "
            "رابطه قابل استفاده سرعت پلیمریزاسیون را ایجاد می‌کند."
        ),
        "formula_latex": r"R_p=k_p[M]\left(\frac{R_i}{2k_t}\right)^{1/2}",
        "variables_json": json.dumps(
            {
                "Rp": "سرعت پلیمریزاسیون",
                "kp": "ثابت سرعت رشد",
                "[M]": "غلظت مونومر",
                "Ri": "سرعت آغازش مؤثر",
                "kt": "ثابت سرعت اختتام",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": (
            "حالت پایا برقرار باشد، اختتام دومولکولی باشد و سرعت مصرف "
            "مونومر در آغازش قابل صرف‌نظرکردن باشد."
        ),
        "cause_fa": (
            "رابطه غلظت رادیکال حالت پایا در Rp = kp[M][M•] جایگذاری می‌شود."
        ),
        "effect_fa": (
            "Rp با [M] و kp رابطه مستقیم، با Ri رابطه ریشه دوم و "
            "با kt رابطه معکوس ریشه دوم دارد."
        ),
        "common_error_fa": (
            "فرض‌کردن وابستگی خطی Rp به Ri یا فراموش‌کردن ضریب 2."
        ),
        "keywords_fa": "رابطه سرعت، سرعت آغازش، ثابت رشد، ثابت اختتام",
        "keywords_en": "rate expression, initiation rate, propagation constant, termination constant",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_009",
        "knowledge_type": "effect_prediction",
        "statement_fa": (
            "دو برابرشدن سرعت آغازش، در شرایط ثابت دیگر، سرعت "
            "پلیمریزاسیون را فقط به اندازه ریشه دوم دو افزایش می‌دهد."
        ),
        "formula_latex": r"\frac{R_{p,2}}{R_{p,1}}=\sqrt{\frac{R_{i,2}}{R_{i,1}}}=\sqrt{2}",
        "variables_json": json.dumps(
            {
                "Rp,1": "سرعت پلیمریزاسیون اولیه",
                "Rp,2": "سرعت پلیمریزاسیون جدید",
                "Ri,1": "سرعت آغازش اولیه",
                "Ri,2": "سرعت آغازش جدید",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": (
            "غلظت مونومر، kp، kt و سایر شرایط ثابت باشند و حالت پایا برقرار باشد."
        ),
        "cause_fa": (
            "افزایش غلظت رادیکال‌ها هم‌زمان سرعت اختتام دومولکولی را افزایش می‌دهد."
        ),
        "effect_fa": (
            "افزایش Ri بازده افزایشی خطی در Rp ایجاد نمی‌کند."
        ),
        "common_error_fa": (
            "نتیجه‌گیری اینکه دو برابرشدن Ri، مقدار Rp را نیز دو برابر می‌کند."
        ),
        "keywords_fa": "وابستگی ریشه دوم، دو برابرشدن سرعت آغازش",
        "keywords_en": "square-root dependence, doubling initiation rate",
    },
    {
        **COMMON,
        "record_id": "odian_3_3b_010",
        "knowledge_type": "timescale",
        "statement_fa": (
            "در بسیاری از پلیمریزاسیون‌های معمول، حالت پایا پس از یک "
            "دوره بسیار کوتاه برقرار می‌شود که حداکثر می‌تواند در حدود "
            "یک دقیقه باشد."
        ),
        "formula_latex": "",
        "variables_json": json.dumps(
            {
                "steady_state_time": "زمان رسیدن تقریبی سیستم به حالت پایا",
            },
            ensure_ascii=False,
        ),
        "assumptions_fa": "این توصیف مربوط به پلیمریزاسیون‌های معمول گزارش‌شده است.",
        "cause_fa": (
            "رادیکال‌های بسیار فعال به‌سرعت میان تشکیل و اختتام به تعادل سینتیکی می‌رسند."
        ),
        "effect_fa": (
            "در بخش عمده زمان واکنش می‌توان از رابطه حالت پایا استفاده کرد."
        ),
        "common_error_fa": (
            "استفاده بدون بررسی از فرض حالت پایا در همان لحظه آغاز واکنش "
            "یا در شرایط غیرپایای ویژه."
        ),
        "keywords_fa": "زمان رسیدن به حالت پایا، دوره آغازین",
        "keywords_en": "steady-state time, initial transient period",
    },
]


def validate(records: list[dict]) -> None:
    errors = []

    if len(records) != 10:
        errors.append(
            f"تعداد رکوردها باید ۱۰ باشد، اما {len(records)} است."
        )

    ids = [record["record_id"] for record in records]

    if len(ids) != len(set(ids)):
        errors.append("شناسه تکراری وجود دارد.")

    expected_ids = {
        f"odian_3_3b_{number:03d}"
        for number in range(1, 11)
    }

    if set(ids) != expected_ids:
        errors.append(
            "شناسه‌ها باید از odian_3_3b_001 تا odian_3_3b_010 باشند."
        )

    for record in records:
        record_id = record["record_id"]

        missing_fields = [
            field
            for field in FIELDNAMES
            if field not in record
        ]

        if missing_fields:
            errors.append(
                f"{record_id}: ستون‌های مفقود: {missing_fields}"
            )
            continue

        if not record["statement_fa"].strip():
            errors.append(
                f"{record_id}: statement_fa خالی است."
            )

        if record["verification_status"] != "source_reviewed":
            errors.append(
                f"{record_id}: وضعیت بررسی نامعتبر است."
            )

        try:
            variables = json.loads(
                record["variables_json"]
            )
        except json.JSONDecodeError:
            errors.append(
                f"{record_id}: variables_json معتبر نیست."
            )
        else:
            if not isinstance(variables, dict):
                errors.append(
                    f"{record_id}: variables_json باید شیء JSON باشد."
                )

        if not (
            206
            <= record["printed_page_start"]
            <= record["printed_page_end"]
            <= 207
        ):
            errors.append(
                f"{record_id}: بازه صفحه خارج از ۲۰۶ تا ۲۰۷ است."
            )

    required_formulas = [
        r"R_p=-\frac{d[M]}{dt}",
        r"R_p=k_p[M][M^\bullet]",
        r"R_t=2k_t[M^\bullet]^2",
        r"[M^\bullet]=\left(\frac{R_i}{2k_t}\right)^{1/2}",
        r"R_p=k_p[M]\left(\frac{R_i}{2k_t}\right)^{1/2}",
    ]

    formulas = {
        record["formula_latex"]
        for record in records
    }

    for formula in required_formulas:
        if formula not in formulas:
            errors.append(
                f"رابطه ضروری پیدا نشد: {formula}"
            )

    if errors:
        print("اعتبارسنجی ناموفق بود:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)


def write_csv(records: list[dict]) -> None:
    with CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(records)


def write_jsonl(records: list[dict]) -> None:
    with JSONL_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate(RECORDS)
    write_csv(RECORDS)
    write_jsonl(RECORDS)

    print("=" * 72)
    print("Knowledge Table بخش 3-3b ساخته شد.")
    print("CSV:", CSV_PATH)
    print("JSONL:", JSONL_PATH)
    print("تعداد رکوردها:", len(RECORDS))
    print("بازه صفحات چاپی: 206 تا 207")
    print("وضعیت: source_reviewed")


if __name__ == "__main__":
    main()
