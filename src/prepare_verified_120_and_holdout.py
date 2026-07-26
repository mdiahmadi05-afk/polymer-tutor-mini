import json
from collections import Counter
from pathlib import Path


INPUT_PATHS = [
    Path(
        "data/training/main/batch_001/"
        "chain_transfer_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_002/"
        "molecular_weight_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_003/"
        "thermal_behavior_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_004/"
        "core_polymer_concepts_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_005/"
        "generalization_verified_fa.jsonl"
    ),
    Path(
        "data/training/main/batch_006/"
        "source_grounding_verified_fa.jsonl"
    ),
]

OUTPUT_DIR = Path(
    "data/training/verified_120"
)

TRAINING_POOL_PATH = (
    OUTPUT_DIR / "polymer_tutor_verified_120.jsonl"
)

HOLDOUT_PATH = (
    OUTPUT_DIR / "holdout_new_15.jsonl"
)

SYSTEM_PROMPT = (
    "تو یک مدرس دقیق علوم پلیمر هستی. "
    "فقط براساس متن منبع ارائه‌شده پاسخ بده. "
    "نمادهای علمی را حفظ کن و پاسخ را مستقیم، "
    "کامل و بدون ادعای ساختگی بنویس. "
    "اگر متن منبع اطلاعات لازم را ندارد، "
    "صریحاً اعلام کن که منبع کافی نیست."
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"فایل پیدا نشد: {path}"
        )

    samples = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSON نامعتبر در {path}، "
                    f"خط {line_number}: {error}"
                ) from error

            samples.append(sample)

    return samples


def make_holdout(
    sample_id: str,
    context: str,
    question: str,
    answer: str,
    subtopic: str,
    sample_type: str,
    source_book: str,
    source_author: str,
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
                    f"متن منبع:\n{context}\n\n"
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
            "topic": "polymer_science",
            "subtopic": subtopic,
            "type": sample_type,
            "difficulty": "intermediate",
            "language": "fa",
            "verified": True,
            "split": "holdout",
            "source": {
                "book": source_book,
                "author": source_author,
                "edition": "Fourth Edition",
                "printed_pages": pages,
            },
        },
    }


ODIAN = "Principles of Polymerization"
ODIAN_AUTHOR = "George Odian"

SPERLING = "Introduction to Physical Polymer Science"
SPERLING_AUTHOR = "L. H. Sperling"


holdout_samples = [
    make_holdout(
        "holdout_001",
        (
            "انتقال زنجیر رشد زنجیر فعال را متوقف می‌کند. "
            "در صورت شروع مجدد سریع، مصرف مونومر تقریباً با "
            "همان سرعت ادامه می‌یابد، ولی زنجیرهای بیشتری "
            "با طول متوسط کمتر تشکیل می‌شوند."
        ),
        (
            "دانشجویی می‌گوید در شروع مجدد سریع، "
            "هم Rp و هم Xn ثابت می‌مانند. نظر او را بررسی کن."
        ),
        (
            "این گفته نادرست است. سرعت پلیمریزاسیون Rp "
            "تقریباً ثابت می‌ماند، اما Xn و وزن مولکولی "
            "کاهش می‌یابند؛ زیرا تعداد بیشتری زنجیر "
            "کوتاه‌تر تشکیل می‌شود."
        ),
        "chain_transfer",
        "error_correction",
        ODIAN,
        ODIAN_AUTHOR,
        [238, 239],
    ),
    make_holdout(
        "holdout_002",
        (
            "ثابت انتقال با رابطه C = ktr/kp تعریف می‌شود."
        ),
        (
            "اگر ktr = 0.015 و kp = 500 باشد، "
            "C را محاسبه کن."
        ),
        (
            "C = ktr/kp = 0.015/500 = 0.00003 "
            "= 3×10^-5 است."
        ),
        "chain_transfer_constant",
        "calculation",
        ODIAN,
        ODIAN_AUTHOR,
        [240, 245],
    ),
    make_holdout(
        "holdout_003",
        (
            "در پلیمریزاسیون مرحله‌ای، وزن مولکولی زیاد "
            "فقط در تبدیل بسیار بالا حاصل می‌شود."
        ),
        (
            "آیا افزایش زمان واکنش در تبدیل پایین برای تولید "
            "پلیمر مرحله‌ای با وزن مولکولی بسیار زیاد کافی است؟"
        ),
        (
            "خیر. عامل اصلی رسیدن به وزن مولکولی زیاد، "
            "دستیابی به تبدیل بسیار بالای گروه‌های عاملی است. "
            "در تبدیل پایین، مقدار زیادی مونومر و الیگومر "
            "باقی می‌ماند."
        ),
        "step_polymerization",
        "explanation",
        ODIAN,
        ODIAN_AUTHOR,
        [39, 50],
    ),
    make_holdout(
        "holdout_004",
        (
            "در پلیمریزاسیون امولسیونی آغازگر معمولاً در آب "
            "محلول است و واکنش در ذرات پلیمری رخ می‌دهد. "
            "در سوسپانسیونی، قطرات مونومر محل اصلی واکنش‌اند."
        ),
        (
            "کدام روش را می‌توان مجموعه‌ای از راکتورهای "
            "توده‌ای کوچک در نظر گرفت و چرا؟"
        ),
        (
            "پلیمریزاسیون سوسپانسیونی؛ زیرا هر قطره مونومر "
            "مانند یک راکتور توده‌ای کوچک عمل می‌کند و "
            "پلیمریزاسیون درون همان قطره انجام می‌شود."
        ),
        "polymerization_processes",
        "explanation",
        ODIAN,
        ODIAN_AUTHOR,
        [350, 353],
    ),
    make_holdout(
        "holdout_005",
        (
            "Mn = ΣNiMi/ΣNi و "
            "Mw = ΣNiMi²/ΣNiMi است."
        ),
        (
            "نمونه‌ای شامل یک زنجیر 10000 و سه زنجیر "
            "30000 گرم بر مول است. Mn و Mw را حساب کن."
        ),
        (
            "Mn = (1×10000 + 3×30000)/4 "
            "= 25000 گرم بر مول. "
            "Mw = (1×10000² + 3×30000²)"
            "/(1×10000 + 3×30000) "
            "= 28000 گرم بر مول."
        ),
        "molecular_weight",
        "calculation",
        SPERLING,
        SPERLING_AUTHOR,
        [85, 86],
    ),
    make_holdout(
        "holdout_006",
        (
            "شاخص پراکندگی از رابطه Đ = Mw/Mn به دست می‌آید."
        ),
        (
            "برای نمونه‌ای با Mn = 60000 و Mw = 72000، "
            "Đ را محاسبه و تفسیر کن."
        ),
        (
            "Đ = 72000/60000 = 1.2 است. این مقدار نسبتاً "
            "نزدیک به 1 است و نشان می‌دهد توزیع وزن مولکولی "
            "نسبتاً باریک است."
        ),
        "dispersity",
        "calculation",
        SPERLING,
        SPERLING_AUTHOR,
        [106, 108],
    ),
    make_holdout(
        "holdout_007",
        (
            "در رابطه مارک–هووینک، "
            "[η] = K Mv^a است."
        ),
        (
            "اگر K = 0.0005، a = 0.5 و Mv = 40000 باشد، "
            "[η] را محاسبه کن."
        ),
        (
            "ریشه دوم 40000 برابر 200 است. بنابراین "
            "[η] = 0.0005×200 = 0.1 دسی‌لیتر بر گرم."
        ),
        "intrinsic_viscosity",
        "calculation",
        SPERLING,
        SPERLING_AUTHOR,
        [110, 115],
    ),
    make_holdout(
        "holdout_008",
        (
            "Tg به تحرک بخش آمورف مربوط است. "
            "Tm دمای ذوب نواحی بلوری است."
        ),
        (
            "پلیمر کاملاً آمورف کدام‌یک از Tg و Tm را "
            "می‌تواند نشان دهد؟"
        ),
        (
            "پلیمر کاملاً آمورف می‌تواند Tg داشته باشد، "
            "اما چون ناحیه بلوری ندارد، Tm بلوری مشخصی "
            "نشان نمی‌دهد."
        ),
        "thermal_behavior",
        "effect_prediction",
        SPERLING,
        SPERLING_AUTHOR,
        [198, 239],
    ),
    make_holdout(
        "holdout_009",
        (
            "ترموست پس از پخت دارای شبکه اتصالات عرضی "
            "دائمی است."
        ),
        (
            "چرا گرم‌کردن یک ترموست پخت‌شده باعث قالب‌گیری "
            "مجدد آن مانند ترموپلاستیک نمی‌شود؟"
        ),
        (
            "زیرا شبکه شیمیایی دائمی مانع لغزش و جریان مستقل "
            "زنجیرها می‌شود. ترموست پخت‌شده ذوب و جاری نمی‌شود "
            "و در دمای زیاد سرانجام تخریب می‌شود."
        ),
        "thermosets",
        "explanation",
        SPERLING,
        SPERLING_AUTHOR,
        [360, 764],
    ),
    make_holdout(
        "holdout_010",
        (
            "در آزمایش خزش، تنش ثابت و کرنش تابع زمان است. "
            "در آرامش تنش، کرنش ثابت و تنش تابع زمان است."
        ),
        (
            "برای بررسی افزایش تدریجی تغییر شکل تحت بار ثابت، "
            "کدام آزمایش مناسب است؟"
        ),
        (
            "آزمایش خزش مناسب است؛ زیرا تنش ثابت نگه داشته "
            "می‌شود و افزایش کرنش با زمان اندازه‌گیری می‌شود."
        ),
        "viscoelasticity",
        "application",
        SPERLING,
        SPERLING_AUTHOR,
        [508, 513],
    ),
    make_holdout(
        "holdout_011",
        (
            "HDPE شاخه‌های کمتری دارد و منظم‌تر بسته‌بندی "
            "می‌شود. LDPE شاخه‌های بیشتری دارد."
        ),
        (
            "کدام‌یک معمولاً چگالی و بلورینگی بیشتری دارد؟ "
            "دلیل را هم بگو."
        ),
        (
            "HDPE معمولاً چگالی و بلورینگی بیشتری دارد، "
            "زیرا شاخه‌های کمتر امکان نزدیک‌شدن و "
            "بسته‌بندی منظم‌تر زنجیرها را فراهم می‌کنند."
        ),
        "polyolefins",
        "structure_property",
        SPERLING,
        SPERLING_AUTHOR,
        [757, 770],
    ),
    make_holdout(
        "holdout_012",
        (
            "افزایش χ نامطلوب، انرژی آزاد اختلاط را افزایش "
            "می‌دهد و امتزاج‌پذیری را کاهش می‌دهد."
        ),
        (
            "با افزایش χ چه تغییری در احتمال جدایش فازی "
            "ایجاد می‌شود؟"
        ),
        (
            "احتمال جدایش فازی افزایش می‌یابد، زیرا "
            "برهم‌کنش‌های اختلاط نامطلوب‌تر و امتزاج‌پذیری "
            "کمتر می‌شود."
        ),
        "phase_separation",
        "effect_prediction",
        SPERLING,
        SPERLING_AUTHOR,
        [148, 155],
    ),
    make_holdout(
        "holdout_013",
        (
            "Mn به هر مولکول سهم عددی برابر می‌دهد، "
            "اما Mw به زنجیرهای سنگین‌تر سهم بیشتری می‌دهد."
        ),
        (
            "اثر افزایش نرم‌کننده بر Tg چیست؟"
        ),
        (
            "متن منبع برای پاسخ به این سؤال کافی نیست؛ "
            "زیرا فقط درباره میانگین‌های وزن مولکولی توضیح می‌دهد."
        ),
        "source_grounding",
        "insufficient_context",
        SPERLING,
        SPERLING_AUTHOR,
        [85, 86],
    ),
    make_holdout(
        "holdout_014",
        (
            "در پلیمریزاسیون سوسپانسیونی، هر قطره مونومر "
            "مانند یک راکتور کوچک عمل می‌کند."
        ),
        (
            "مقدار شاخص پراکندگی Đ چگونه محاسبه می‌شود؟"
        ),
        (
            "متن منبع برای محاسبه شاخص پراکندگی کافی نیست. "
            "رابطه مربوط به Mn و Mw در این متن ارائه نشده است."
        ),
        "source_grounding",
        "insufficient_context",
        ODIAN,
        ODIAN_AUTHOR,
        [350, 353],
    ),
    make_holdout(
        "holdout_015",
        (
            "Tg به افزایش تحرک بخش آمورف پلیمر مربوط است."
        ),
        (
            "نسبت واکنش‌پذیری r1 در کوپلیمریزاسیون "
            "چگونه تعریف می‌شود؟"
        ),
        (
            "متن منبع برای پاسخ کافی نیست؛ زیرا درباره "
            "گذار شیشه‌ای است و اطلاعاتی درباره نسبت "
            "واکنش‌پذیری کوپلیمریزاسیون ارائه نمی‌کند."
        ),
        "source_grounding",
        "insufficient_context",
        SPERLING,
        SPERLING_AUTHOR,
        [239, 349],
    ),
]


def validate_messages(
    sample: dict,
    expected_split: str | None = None,
) -> list[str]:
    errors = []
    metadata = sample.get("metadata", {})
    sample_id = metadata.get("id")
    messages = sample.get("messages")

    if not isinstance(messages, list) or len(messages) != 3:
        return [
            f"{sample_id}: messages باید دقیقاً ۳ عضو داشته باشد."
        ]

    roles = [
        message.get("role")
        for message in messages
    ]

    if roles != ["system", "user", "assistant"]:
        errors.append(
            f"{sample_id}: ترتیب نقش‌ها نامعتبر است."
        )

    for message in messages:
        content = message.get("content")

        if not isinstance(content, str) or not content.strip():
            errors.append(
                f"{sample_id}: پیام خالی وجود دارد."
            )

    if metadata.get("verified") is not True:
        errors.append(
            f"{sample_id}: verified باید True باشد."
        )

    if metadata.get("language") != "fa":
        errors.append(
            f"{sample_id}: language باید fa باشد."
        )

    if (
        expected_split is not None
        and metadata.get("split") != expected_split
    ):
        errors.append(
            f"{sample_id}: split باید {expected_split} باشد."
        )

    return errors


def write_jsonl(
    path: Path,
    samples: list[dict],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(
                json.dumps(
                    sample,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    training_samples = []
    training_ids = set()
    errors = []

    for path in INPUT_PATHS:
        samples = load_jsonl(path)

        print(f"{path}: {len(samples)} نمونه")

        for sample in samples:
            errors.extend(
                validate_messages(sample)
            )

            sample_id = sample["metadata"]["id"]

            if sample_id in training_ids:
                errors.append(
                    f"شناسه تکراری در Training: {sample_id}"
                )

            training_ids.add(sample_id)
            training_samples.append(sample)

    if len(training_samples) != 120:
        errors.append(
            f"تعداد Training باید ۱۲۰ باشد، "
            f"اما {len(training_samples)} است."
        )

    holdout_ids = set()

    for sample in holdout_samples:
        errors.extend(
            validate_messages(
                sample,
                expected_split="holdout",
            )
        )

        sample_id = sample["metadata"]["id"]

        if sample_id in holdout_ids:
            errors.append(
                f"شناسه تکراری در Holdout: {sample_id}"
            )

        holdout_ids.add(sample_id)

    if len(holdout_samples) != 15:
        errors.append(
            f"تعداد Holdout باید ۱۵ باشد، "
            f"اما {len(holdout_samples)} است."
        )

    overlap = training_ids & holdout_ids

    if overlap:
        errors.append(
            f"هم‌پوشانی شناسه بین Training و Holdout: "
            f"{sorted(overlap)}"
        )

    training_questions = {
        sample["messages"][1]["content"]
        for sample in training_samples
    }

    holdout_questions = {
        sample["messages"][1]["content"]
        for sample in holdout_samples
    }

    prompt_overlap = (
        training_questions & holdout_questions
    )

    if prompt_overlap:
        errors.append(
            "حداقل یک پیام کاربر میان Training و Holdout "
            "کاملاً تکراری است."
        )

    if errors:
        print("\nاعتبارسنجی ناموفق بود:")

        for error in errors:
            print("-", error)

        raise SystemExit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        TRAINING_POOL_PATH,
        training_samples,
    )

    write_jsonl(
        HOLDOUT_PATH,
        holdout_samples,
    )

    batch_counts = Counter(
        sample["metadata"]["id"].split("_")[0]
        for sample in training_samples
    )

    holdout_type_counts = Counter(
        sample["metadata"]["type"]
        for sample in holdout_samples
    )

    print("\n" + "=" * 72)
    print("فایل Training:")
    print(TRAINING_POOL_PATH)
    print("تعداد نمونه‌های Training:", len(training_samples))

    print("\nفایل Holdout:")
    print(HOLDOUT_PATH)
    print("تعداد نمونه‌های Holdout:", len(holdout_samples))

    print("\nهم‌پوشانی شناسه‌ها:", len(overlap))
    print(
        "تکرار کامل پیام کاربر:",
        len(prompt_overlap),
    )

    print("\nتوزیع پیشوند شناسه‌های Training:")

    for prefix, count in sorted(batch_counts.items()):
        print(f"- {prefix}: {count}")

    print("\nتوزیع انواع Holdout:")

    for sample_type, count in sorted(
        holdout_type_counts.items()
    ):
        print(f"- {sample_type}: {count}")

    print("\nنتیجه:")
    print("دیتاست ۱۲۰تایی با موفقیت ادغام شد.")
    print("Holdout جدید ۱۵تایی ساخته شد.")
    print("Holdout وارد آموزش نشده است.")


if __name__ == "__main__":
    main()
