import json
from pathlib import Path

DATASET_PATH = Path("data/training/pilot_train_fa.jsonl")

SYSTEM_PROMPT = (
    "تو یک مدرس دقیق علوم و مهندسی پلیمر هستی. "
    "پاسخ را مستقیم، علمی، روان و بدون مقدمه اضافی ارائه کن."
)

new_samples = [
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "خطی یا شاخه‌دار بودن زنجیر پلیمر چه اثری بر بسته‌بندی زنجیرها دارد؟",
            },
            {
                "role": "assistant",
                "content": (
                    "زنجیرهای خطی معمولاً می‌توانند نزدیک‌تر و منظم‌تر کنار هم قرار بگیرند؛ "
                    "بنابراین امکان تشکیل نواحی بلوری و افزایش چگالی بیشتر است. شاخه‌ها، به‌ویژه "
                    "شاخه‌های زیاد یا بلند، نظم فضایی زنجیرها را مختل می‌کنند و مانع بسته‌بندی "
                    "فشرده می‌شوند. در نتیجه بلورینگی و چگالی معمولاً کاهش می‌یابد."
                ),
            },
        ],
        "metadata": {
            "topic": "chain_structure",
            "type": "effect_prediction",
            "difficulty": "basic",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "چرا پلیمر ایزوتاکتیک معمولاً آسان‌تر از پلیمر آتاکتیک متبلور می‌شود؟",
            },
            {
                "role": "assistant",
                "content": (
                    "در پلیمر ایزوتاکتیک، گروه‌های جانبی آرایش منظم و تکرارشونده‌ای دارند. "
                    "این نظم فضایی باعث می‌شود زنجیرها بهتر در کنار یکدیگر قرار بگیرند و ساختار "
                    "بلوری تشکیل دهند. در پلیمر آتاکتیک، آرایش گروه‌های جانبی نامنظم است و "
                    "بسته‌بندی منظم زنجیرها دشوارتر می‌شود؛ بنابراین ماده معمولاً آمورف‌تر است."
                ),
            },
        ],
        "metadata": {
            "topic": "chain_structure",
            "type": "explanation",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "تفاوت رفتار پلیمر در دمای انتقال شیشه‌ای و دمای ذوب چیست؟",
            },
            {
                "role": "assistant",
                "content": (
                    "در دمای انتقال شیشه‌ای، حرکت سگمنتی بخش‌های آمورف زنجیر افزایش می‌یابد و "
                    "پلیمر از حالت شیشه‌ای و سخت به حالت نرم‌تر و لاستیکی‌تر می‌رود. این پدیده "
                    "ذوب‌شدن نیست. دمای ذوب مربوط به ازبین‌رفتن نظم نواحی بلوری است. بنابراین "
                    "Tg در بخش آمورف مشاهده می‌شود، اما Tm فقط در پلیمرهای دارای نواحی بلوری وجود دارد."
                ),
            },
        ],
        "metadata": {
            "topic": "thermal_behavior",
            "type": "comparison",
            "difficulty": "basic",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "چرا یک پلیمر گرماسخت هنگام گرم‌شدن مانند ترموپلاستیک جاری نمی‌شود؟",
            },
            {
                "role": "assistant",
                "content": (
                    "پلیمر گرماسخت دارای شبکه‌ای از اتصالات عرضی کووالانسی است. این اتصالات "
                    "مانع حرکت مستقل و لغزش آزاد زنجیرها می‌شوند. به همین دلیل گرماسخت با "
                    "گرم‌شدن به مذاب قابل‌جریان تبدیل نمی‌شود و در دمای زیاد معمولاً تخریب "
                    "شیمیایی رخ می‌دهد. ترموپلاستیک‌ها شبکه دائمی ندارند و می‌توانند با گرم‌شدن نرم یا مذاب شوند."
                ),
            },
        ],
        "metadata": {
            "topic": "polymer_basics",
            "type": "mechanism",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "افزایش چگالی اتصالات عرضی چه اثری بر مدول و ازدیاد طول پلیمر دارد؟",
            },
            {
                "role": "assistant",
                "content": (
                    "افزایش چگالی اتصالات عرضی حرکت زنجیرها را محدود می‌کند. در نتیجه مدول و "
                    "سفتی معمولاً افزایش می‌یابد، اما توانایی زنجیرها برای بازشدن و تغییر شکل "
                    "کاهش پیدا می‌کند؛ بنابراین ازدیاد طول در شکست و انعطاف‌پذیری معمولاً کمتر می‌شود. "
                    "اتصال عرضی بیش‌ازحد می‌تواند ماده را شکننده کند."
                ),
            },
        ],
        "metadata": {
            "topic": "structure_property",
            "type": "effect_prediction",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "نمونه‌ای شامل ۱۰۰ زنجیر با جرم مولکولی ۱۰۰۰۰ و ۵۰ زنجیر با جرم مولکولی ۲۰۰۰۰ است. Mn را محاسبه کن.",
            },
            {
                "role": "assistant",
                "content": (
                    "برای محاسبه متوسط عددی از رابطه Mn = مجموع NiMi تقسیم بر مجموع Ni استفاده می‌کنیم. "
                    "صورت کسر برابر است با: ۱۰۰×۱۰۰۰۰ + ۵۰×۲۰۰۰۰ = ۲۰۰۰۰۰۰. "
                    "تعداد کل زنجیرها برابر ۱۵۰ است. بنابراین Mn = ۲۰۰۰۰۰۰÷۱۵۰، "
                    "یعنی تقریباً ۱۳۳۳۳ گرم بر مول."
                ),
            },
        ],
        "metadata": {
            "topic": "molecular_weight",
            "type": "step_by_step_calculation",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "برای پلیمری با Mw برابر ۱۲۰۰۰۰ و Mn برابر ۶۰۰۰۰، پراکندگی را محاسبه و تفسیر کن.",
            },
            {
                "role": "assistant",
                "content": (
                    "پراکندگی از نسبت Mw به Mn به دست می‌آید. بنابراین Đ = ۱۲۰۰۰۰÷۶۰۰۰۰ = ۲. "
                    "این مقدار نشان می‌دهد نمونه دارای توزیع وزن مولکولی است و همه زنجیرها جرم "
                    "مولکولی یکسانی ندارند. هرچه مقدار پراکندگی به ۱ نزدیک‌تر باشد، توزیع باریک‌تر است."
                ),
            },
        ],
        "metadata": {
            "topic": "molecular_weight",
            "type": "step_by_step_calculation",
            "difficulty": "basic",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "اگر Mn یک پلیمر ۵۶۰۰۰ و جرم مولی واحد تکرارشونده ۲۸ باشد، درجه پلیمریزاسیون عددی چقدر است؟",
            },
            {
                "role": "assistant",
                "content": (
                    "در تقریب ساده، درجه پلیمریزاسیون عددی از تقسیم Mn بر جرم مولی واحد "
                    "تکرارشونده به دست می‌آید. بنابراین DPn = ۵۶۰۰۰÷۲۸ = ۲۰۰۰. "
                    "یعنی هر زنجیر به‌طور متوسط حدود ۲۰۰۰ واحد تکرارشونده دارد."
                ),
            },
        ],
        "metadata": {
            "topic": "molecular_weight",
            "type": "step_by_step_calculation",
            "difficulty": "basic",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "طبق رابطه مارک–هوینک، افزایش وزن مولکولی چه اثری بر ویسکوزیته ذاتی دارد؟",
            },
            {
                "role": "assistant",
                "content": (
                    "رابطه مارک–هوینک به صورت [η] = K M^a نوشته می‌شود. برای یک سامانه مشخص "
                    "پلیمر–حلال و دمای ثابت، K و a ثابت‌اند و a معمولاً مثبت است. بنابراین با "
                    "افزایش وزن مولکولی، ویسکوزیته ذاتی افزایش می‌یابد. مقدار دقیق افزایش به توان a وابسته است."
                ),
            },
        ],
        "metadata": {
            "topic": "molecular_weight",
            "type": "effect_prediction",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "چرا برای بسیاری از پلیمرها نقطه جوش معمولی تعریف نمی‌شود؟",
            },
            {
                "role": "assistant",
                "content": (
                    "پلیمرها مولکول‌های بسیار بزرگی دارند و نیروهای بین‌مولکولی کل آن‌ها زیاد است. "
                    "دمای لازم برای جداکردن کامل این مولکول‌ها و ورودشان به فاز بخار معمولاً آن‌قدر "
                    "بالاست که پیوندهای شیمیایی زنجیر پیش از جوشیدن شکسته می‌شوند. بنابراین بیشتر "
                    "پلیمرها ابتدا تجزیه حرارتی می‌شوند و نقطه جوش مشخصی ندارند."
                ),
            },
        ],
        "metadata": {
            "topic": "polymer_basics",
            "type": "explanation",
            "difficulty": "basic",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "کوپلیمرهای تصادفی، متناوب، بلوکی و پیوندی را به‌طور خلاصه مقایسه کن.",
            },
            {
                "role": "assistant",
                "content": (
                    "در کوپلیمر تصادفی، واحدهای دو مونومر بدون نظم طولانی مشخص در زنجیر پخش می‌شوند. "
                    "در کوپلیمر متناوب، دو واحد به‌صورت منظم یکی‌درمیان قرار می‌گیرند. در کوپلیمر "
                    "بلوکی، بخش‌های نسبتاً بلند از یک مونومر به بخش‌های مونومر دیگر متصل‌اند. "
                    "در کوپلیمر پیوندی، زنجیرهای جانبی یک پلیمر به ستون فقرات پلیمر دیگری متصل می‌شوند."
                ),
            },
        ],
        "metadata": {
            "topic": "polymer_basics",
            "type": "comparison",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "وجود گروه جانبی حجیم معمولاً چه اثری بر Tg پلیمر دارد و چرا؟",
            },
            {
                "role": "assistant",
                "content": (
                    "گروه جانبی حجیم معمولاً چرخش پیوندها و حرکت سگمنتی زنجیر را محدود می‌کند. "
                    "در نتیجه برای آغاز حرکت هماهنگ بخش‌های زنجیر، دمای بیشتری لازم است و Tg "
                    "معمولاً افزایش می‌یابد. بااین‌حال اثر نهایی به انعطاف‌پذیری گروه جانبی، "
                    "برهم‌کنش‌های بین‌مولکولی و ساختار کلی زنجیر نیز وابسته است."
                ),
            },
        ],
        "metadata": {
            "topic": "thermal_behavior",
            "type": "effect_prediction",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "چرا بیشتر پلیمرهای بلورین در عمل کاملاً بلوری نیستند؟",
            },
            {
                "role": "assistant",
                "content": (
                    "زنجیرهای پلیمر بسیار بلند، درهم‌تنیده و دارای نقص‌های ساختاری‌اند و نمی‌توانند "
                    "در تمام حجم ماده به‌طور کاملاً منظم بسته‌بندی شوند. شاخه‌ها، انتهای زنجیرها، "
                    "توزیع وزن مولکولی و سرعت سردشدن نیز نظم را مختل می‌کنند. بنابراین معمولاً "
                    "نواحی بلوری و آمورف هم‌زمان وجود دارند و ماده نیمه‌بلوری است."
                ),
            },
        ],
        "metadata": {
            "topic": "morphology",
            "type": "explanation",
            "difficulty": "intermediate",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "اثر نرم‌کننده و اتصال عرضی بر حرکت زنجیرها را مقایسه کن.",
            },
            {
                "role": "assistant",
                "content": (
                    "نرم‌کننده معمولاً حجم آزاد را افزایش می‌دهد و حرکت سگمنتی زنجیرها را آسان‌تر "
                    "می‌کند؛ بنابراین انعطاف‌پذیری بیشتر و Tg کمتر می‌شود. اتصال عرضی زنجیرها را "
                    "به یکدیگر متصل و حرکت آن‌ها را محدود می‌کند؛ بنابراین سفتی و پایداری ابعادی "
                    "بیشتر می‌شود و انعطاف‌پذیری معمولاً کاهش می‌یابد."
                ),
            },
        ],
        "metadata": {
            "topic": "structure_property",
            "type": "comparison",
            "difficulty": "basic",
        },
    },
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "این ادعا را اصلاح کن: هرچه میزان شاخه‌دارشدن پلی‌اتیلن بیشتر باشد، بلورینگی آن بیشتر می‌شود.",
            },
            {
                "role": "assistant",
                "content": (
                    "این ادعا نادرست است. افزایش شاخه‌دارشدن معمولاً مانع نزدیک‌شدن و بسته‌بندی "
                    "منظم زنجیرهای پلی‌اتیلن می‌شود؛ بنابراین بلورینگی و چگالی کاهش می‌یابد. "
                    "به همین دلیل پلی‌اتیلن کم‌چگالی که شاخه‌های بیشتری دارد، معمولاً نسبت به "
                    "پلی‌اتیلن پرچگالی بلورینگی کمتری دارد."
                ),
            },
        ],
        "metadata": {
            "topic": "structure_property",
            "type": "error_correction",
            "difficulty": "basic",
        },
    },
]

existing_samples = []

if DATASET_PATH.exists():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                existing_samples.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"خط {line_number} فایل دیتاست JSON معتبر نیست: {error}"
                ) from error

existing_questions = {
    sample["messages"][1]["content"] for sample in existing_samples
}

added_count = 0

for sample in new_samples:
    question = sample["messages"][1]["content"]
    if question not in existing_questions:
        existing_samples.append(sample)
        existing_questions.add(question)
        added_count += 1

with DATASET_PATH.open("w", encoding="utf-8") as file:
    for sample in existing_samples:
        file.write(json.dumps(sample, ensure_ascii=False) + "\n")

print(f"نمونه‌های جدید افزوده‌شده: {added_count}")
print(f"تعداد کل نمونه‌ها: {len(existing_samples)}")

if len(existing_samples) != 20:
    raise ValueError(
        f"تعداد مورد انتظار 20 نمونه بود، اما {len(existing_samples)} نمونه وجود دارد."
    )

print("دیتاست آزمایشی 20 نمونه‌ای سالم است.")
