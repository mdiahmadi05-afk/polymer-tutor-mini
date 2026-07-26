import json
from pathlib import Path


OUTPUT_PATH = Path(
    "data/training/main/batch_002/"
    "molecular_weight_verified_fa.jsonl"
)

SYSTEM_PROMPT = (
    "تو یک مدرس دقیق علوم پلیمر هستی. "
    "فقط براساس متن منبع ارائه‌شده پاسخ بده. "
    "بین Mn، Mw، Mv، درجه پلیمریزاسیون و وزن مولکولی "
    "تفاوت دقیق قائل شو. فرمول‌ها، نمادها و واحدها را تغییر نده. "
    "از تعریف‌ها و تفسیرهای ساختگی خودداری کن."
)


def make_sample(
    sample_id: str,
    source_text: str,
    question: str,
    answer: str,
    sample_type: str,
    pages: list[int],
) -> dict:
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    f"متن منبع:\n{source_text}\n\n"
                    f"سؤال:\n{question}"
                ),
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "metadata": {
            "id": sample_id,
            "topic": "physical_polymer_science",
            "subtopic": "molecular_weight_averages",
            "type": sample_type,
            "difficulty": "intermediate",
            "language": "fa",
            "verified": True,
            "source": {
                "book": "Introduction to Physical Polymer Science",
                "author": "L. H. Sperling",
                "edition": "Fourth Edition",
                "printed_pages": pages,
            },
        },
    }


samples = [
    make_sample(
        sample_id="mw_001",
        source_text=(
            "وزن مولکولی متوسط عددی از تقسیم مجموع وزن مولکولی "
            "همه مولکول‌ها بر تعداد کل مولکول‌ها به دست می‌آید. "
            "رابطه آن Mn = ΣNiMi / ΣNi است."
        ),
        question="وزن مولکولی متوسط عددی Mn چیست؟",
        answer=(
            "Mn میانگینی است که در آن هر مولکول، مستقل از اندازه‌اش، "
            "یک سهم عددی برابر دارد. رابطه آن به‌صورت "
            "Mn = ΣNiMi / ΣNi است؛ در این رابطه Ni تعداد مولکول‌های "
            "دارای وزن مولکولی Mi است."
        ),
        sample_type="definition",
        pages=[85, 86],
    ),
    make_sample(
        sample_id="mw_002",
        source_text=(
            "وزن مولکولی متوسط وزنی با رابطه "
            "Mw = ΣNiMi² / ΣNiMi تعریف می‌شود. "
            "مولکول‌های سنگین‌تر سهم بیشتری در Mw دارند."
        ),
        question="وزن مولکولی متوسط وزنی Mw چیست؟",
        answer=(
            "Mw میانگینی است که سهم زنجیرهای سنگین‌تر را بیشتر لحاظ "
            "می‌کند. رابطه آن Mw = ΣNiMi² / ΣNiMi است. بنابراین Mw "
            "نسبت به Mn حساسیت بیشتری به بخش پر‌وزن توزیع دارد."
        ),
        sample_type="definition",
        pages=[85, 86],
    ),
    make_sample(
        sample_id="mw_003",
        source_text=(
            "در یک توزیع وزن مولکولی، معمولاً Mw بزرگ‌تر یا مساوی Mn "
            "است. فقط برای نمونه‌ای که تمام مولکول‌های آن وزن یکسان "
            "دارند، Mw و Mn برابر می‌شوند."
        ),
        question="چرا معمولاً Mw از Mn بزرگ‌تر است؟",
        answer=(
            "زیرا در محاسبه Mw، زنجیرهای سنگین‌تر وزن آماری بیشتری "
            "دارند، اما در Mn هر مولکول یک سهم عددی برابر دارد. "
            "بنابراین معمولاً Mw ≥ Mn است و برابری آن‌ها فقط برای "
            "توزیع کاملاً یکنواخت رخ می‌دهد."
        ),
        sample_type="comparison",
        pages=[85, 86],
    ),
    make_sample(
        sample_id="mw_004",
        source_text=(
            "شاخص پراکندگی یا Dispersity با نسبت Đ = Mw/Mn تعریف "
            "می‌شود. مقدار یک بیانگر نمونه کاملاً یکنواخت است و "
            "افزایش مقدار آن نشان‌دهنده گسترده‌ترشدن توزیع است."
        ),
        question=(
            "شاخص پراکندگی وزن مولکولی چگونه تعریف و تفسیر می‌شود؟"
        ),
        answer=(
            "شاخص پراکندگی از رابطه Đ = Mw/Mn به دست می‌آید. "
            "کمترین مقدار آن 1 است. هرچه Đ از 1 بزرگ‌تر شود، اختلاف "
            "بین اندازه زنجیرها و پهنای توزیع وزن مولکولی بیشتر است. "
            "اصطلاح قدیمی PDI نیز برای این نسبت استفاده می‌شود."
        ),
        sample_type="interpretation",
        pages=[106, 107],
    ),
    make_sample(
        sample_id="mw_005",
        source_text=(
            "برای محاسبه Mn از رابطه Mn = ΣNiMi / ΣNi استفاده می‌شود."
        ),
        question=(
            "نمونه‌ای شامل دو مولکول با وزن مولکولی 10000 و یک "
            "مولکول با وزن مولکولی 40000 گرم بر مول است. Mn را حساب کن."
        ),
        answer=(
            "مجموع جرم مولکول‌ها برابر است با "
            "2×10000 + 1×40000 = 60000. تعداد کل مولکول‌ها 3 است. "
            "بنابراین Mn = 60000/3 = 20000 گرم بر مول."
        ),
        sample_type="calculation",
        pages=[85, 86],
    ),
    make_sample(
        sample_id="mw_006",
        source_text=(
            "برای محاسبه Mw از رابطه "
            "Mw = ΣNiMi² / ΣNiMi استفاده می‌شود."
        ),
        question=(
            "نمونه‌ای شامل دو مولکول با وزن مولکولی 10000 و یک "
            "مولکول با وزن مولکولی 40000 گرم بر مول است. Mw را حساب کن."
        ),
        answer=(
            "صورت کسر برابر است با "
            "2×10000² + 1×40000² = 1800000000. "
            "مخرج برابر است با 2×10000 + 1×40000 = 60000. "
            "بنابراین Mw = 1800000000/60000 = 30000 گرم بر مول."
        ),
        sample_type="calculation",
        pages=[85, 86],
    ),
    make_sample(
        sample_id="mw_007",
        source_text=(
            "ویسکوزیته ذاتی با رابطه مارک–هووینک "
            "[η] = K Mv^a به وزن مولکولی متوسط ویسکوزی مرتبط است. "
            "K و a ثابت‌های تجربی هستند."
        ),
        question=(
            "اگر K = 0.0001 دسی‌لیتر بر گرم، a = 0.8 و "
            "Mv = 100000 گرم بر مول باشد، ویسکوزیته ذاتی را حساب کن."
        ),
        answer=(
            "از رابطه [η] = K Mv^a استفاده می‌کنیم. "
            "داریم 100000^0.8 = 10000. بنابراین "
            "[η] = 0.0001×10000 = 1 دسی‌لیتر بر گرم."
        ),
        sample_type="calculation",
        pages=[110, 111],
    ),
    make_sample(
        sample_id="mw_008",
        source_text=(
            "ثابت‌های K و a در رابطه مارک–هووینک به نوع پلیمر، "
            "نوع حلال و دمای اندازه‌گیری وابسته‌اند."
        ),
        question=(
            "آیا می‌توان ثابت‌های K و a را برای تمام پلیمرها و "
            "حلال‌ها یکسان در نظر گرفت؟"
        ),
        answer=(
            "خیر. K و a ثابت‌های عمومی و یکسان برای همه سامانه‌ها "
            "نیستند. مقدار آن‌ها برای هر جفت پلیمر–حلال و در دمای "
            "مشخص تعیین می‌شود؛ بنابراین تغییر پلیمر، حلال یا دما "
            "می‌تواند مقدار این ثابت‌ها را تغییر دهد."
        ),
        sample_type="error_correction",
        pages=[110, 111],
    ),
    make_sample(
        sample_id="mw_009",
        source_text=(
            "Mv وزن مولکولی متوسطی است که از اندازه‌گیری ویسکوزیته "
            "ذاتی و رابطه مارک–هووینک به دست می‌آید. برای مقادیر "
            "معمول نمای a بین صفر و یک، Mn ≤ Mv ≤ Mw است."
        ),
        question="تفاوت Mv با Mn و Mw چیست؟",
        answer=(
            "Mn از شمارش عددی مولکول‌ها و Mw از وزن‌دهی بیشتر به "
            "زنجیرهای سنگین به دست می‌آید. Mv از رفتار ویسکوزی محلول "
            "و رابطه مارک–هووینک تعیین می‌شود. برای حالت معمول "
            "0 < a < 1، رابطه Mn ≤ Mv ≤ Mw برقرار است."
        ),
        sample_type="comparison",
        pages=[110, 111],
    ),
    make_sample(
        sample_id="mw_010",
        source_text=(
            "Mw مجموع وزن‌های مولکولی تقسیم بر تعداد مولکول‌ها نیست؛ "
            "این تعریف مربوط به Mn است. در Mw جمله Mi به توان دو در "
            "صورت کسر ظاهر می‌شود."
        ),
        question=(
            "این جمله را اصلاح کن: «Mw از تقسیم مجموع وزن مولکولی "
            "همه زنجیرها بر تعداد زنجیرها به دست می‌آید.»"
        ),
        answer=(
            "جمله نادرست است. تقسیم مجموع وزن مولکولی زنجیرها بر "
            "تعداد زنجیرها، Mn را می‌دهد. Mw با رابطه "
            "Mw = ΣNiMi² / ΣNiMi محاسبه می‌شود و به زنجیرهای "
            "سنگین‌تر سهم بیشتری می‌دهد."
        ),
        sample_type="error_correction",
        pages=[85, 86],
    ),
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(sample, ensure_ascii=False) + "\n"
            )

    print("فایل ساخته شد:", OUTPUT_PATH)
    print("تعداد نمونه‌ها:", len(samples))


if __name__ == "__main__":
    main()
