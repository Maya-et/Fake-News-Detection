import pandas as pd
import random

# ==========================================
# 1. DEFINING UNIQUE PATTERNS FOR 500 FAKE NEWS (Label: 1)
# ==========================================
fake_templates = [
    "አስቸኳይ መረጃ፡ በ{city} ከፍተኛ የህዝብ ተቃውሞ በመነሳቱ ሙሉ በሙሉ የሰዓት እላፊ ገደብ መጣሉን የ{media} ገለጹ።",
    "ብሄራዊ የቪዛ ድጋፍ፡ ሁሉም የ{univ} ተማሪዎች ያለ ምንም መስፈርት ወደ አውሮፓ የሚያስጉዝ የነጻ ስኮላርሺፕ እድል ተሰጣቸው።",
    "የኢትዮጵያ ንግድ ባንክ በሲስተም ብልሽት ምክንያት ትላንት ማታ የጠፋውን {money} ሚሊዮን ብር ለማግኘት ለጠቆመ ሰው ሽልማት ሊሰጥ ነው።",
    "በኢንተርኔት ላይ የሚሰራጩ አዳዲስ መተግበሪያዎች የዜጎችን የባንክ ሚስጥር በ{percent}% እየሰረቁ መሆኑን ደህንነቶች በምስጢር አሳወቁ።",
    "የ{utility} መዋቅር በከፍተኛ የሳይበር ጥቃት በመመታቱ በመላው ሀገሪቱ የኤሌክትሪክ አገልግሎት ላልተወሰነ ጊዜ ሊቋረጥ ነው።",
    "ታዋቂው የ{app} ገጽ በሀገር ውስጥ የጸጥታ ስጋት መፍጠሩን ተከትሎ እስከ ፊታችን {day} ድረስ ሙሉ በሙሉ ሊታገድ ነው።",
    "አዲስ መመሪያ፡ ከሚቀጥለው ወር ጀምሮ ማንኛውም ሰው በባንክ አካውንቱ ማስቀመጥ የሚችለው ከፍተኛ የገንዘብ መጠን {money} ብር ብቻ ነው።",
    "የአለም ጤና ድርጅት በ{city} አዲስ ገዳይ በሽታ መከሰቱን ተከትሎ አስቸኳይ የጉዞ እገዳ መጣሉን በምስጢር ገለጸ።",
    "አስደሳች ዜና፡ የ{bank} ደንበኞች በሙሉ ይህንን ሊንክ በመጫን የ{money} ብር ቦነስ በነጻ እንዲወስዱ ተጋብዘዋል።",
    "በሀገር አቀፍ ደረጃ አዲስ የ{money_note} ብር የኖት እትም በምስጢር መታተሙን የኢኮኖሚ ባለሙያዎች አረጋገጡ።"
]

# ==========================================
# 2. DEFINING UNIQUE PATTERNS FOR 500 TRUE NEWS (Label: 0)
# ==========================================
true_templates = [
    "የአዲስ አበባ ከተማ አስተዳደር በ{city} ዙሪያ አዳዲስ የጋራ መኖሪያ ቤቶችን ለመገንባት ከኮንስትራክሽን ማህበራት ጋር ተወያየ።",
    "የትምህርት ሚኒስቴር የ12ኛ ክፍል ብሄራዊ ፈተና በ{univ} ቅጥር ግቢ ውስጥ በሰላም መጠናቀቁን በይፋዊ መግለጫው አስታወቀ።",
    "የኢትዮጵያ ብሄራዊ ባንክ የውጭ ምንዛሪ ተመን ላይ አዲስ የገበያ ማሻሻያ መመሪያ ማውጣቱን የ{media} ዘገበ።",
    "የኢትዮጵያ ኤሌክትሪክ ኃይል በ{city} አካባቢ የተበላሹ የከፍተኛ ኃይል ማስተላለፊያ መስመሮችን ጠገነ።",
    "ኢትዮ ቴሌኮም በ{city} ከተማ የነበረውን የኔትወርክ አቅም ለማሳደግ አዳዲስ የሞባይል ጣቢያዎችን መትከሉን ገለጸ።",
    "የጤና ሚኒስቴር ከዓለም አቀፍ ለጋሽ ድርጅቶች ጋር በመተባበር በ{city} ለሚገኙ ሆስፒታሎች የህክምና ቁሳቁስ ድጋፍ አደረገ።",
    "የኢትዮጵያ ንግድ ባንክ ደንበኞች የዲጂታል ባንክ አገልግሎትን በሰፊው እንዲጠቀሙ አዲስ የሞባይል መተግበሪያ አሻሻለ።",
    "የኢትዮጵያ ብሄራዊ እግር ኳስ ቡድን ለሚቀጥለው የአፍሪካ ዋንጫ ማጣሪያ ዝግጅቱን በ{city} ስታዲየም መጀመሩ ታወቀ።",
    "የግብርና ሚኒስቴር በዘንድሮው የመኸር ወቅት በ{city} ዙሪያ ከ{percent}% በላይ የሚሆነውን የእርሻ መሬት በትራክተር ማረሱን ገለጸ።",
    "በመላው ሀገሪቱ የሚገኙ የከፍተኛ ትምህርት ተቋማት የዘንድሮውን የባህልና የስፖርት ፌስቲቫል በ{univ} ለማክበር ተስማሙ።"
]

# Lexicon items for randomization mapping
univs = ["ወልቂጤ ዩኒቨርሲቲ", "አዲስ አበባ ዩኒቨርሲቲ", "ሀዋሳ ዩኒቨርሲቲ", "ባህር ዳር ዩኒቨርሲቲ", "ጂማ ዩኒቨርሲቲ"]
banks = ["የኢትዮጵያ ንግድ ባንክ", "አዋሽ ባንክ", "አቢሲኒያ ባንክ", "ዳሸን ባንክ", "ኦሮሚያ ባንክ"]
cities = ["ወልቂጤ", "አዲስ አበባ", "ናዝሬት", "ባህር ዳር", "ሀዋሳ", "ድሬዳዋ", "ጎንደር", "ደሴ"]
utilities = ["የመብራት ኃይል", "የውሃ እና ፍሳሽ", "የቴሌኮም ኔትወርክ"]
apps = ["ቲክቶክ (TikTok)", "ቴሌግራም (Telegram)", "ፌስቡክ (Facebook)"]
media = ["ፋና ብሮድካስቲንግ", "የኢትዮጵያ ፕሬስ ድርጅት", "የኢትዮጵያ ዜና አገልግሎት"]
days = ["ሰኞ", "ማክሰኞ", "ረቡዕ", "ሐሙስ", "አርብ"]

all_data = []

# Generate 500 Unique FAKE News rows
fake_set = set()
while len(fake_set) < 500:
    text = random.choice(fake_templates).format(
        univ=random.choice(univs), bank=random.choice(banks), city=random.choice(cities),
        utility=random.choice(utilities), app=random.choice(apps), media=random.choice(media),
        day=random.choice(days), money=random.randint(4000, 25000), percent=random.randint(40, 95),
        money_note=random.choice(["200", "1000"])
    )
    if text not in fake_set:
        fake_set.add(text)
        all_data.append({"text": text, "label": 1})

# Generate 500 Unique TRUE News rows
true_set = set()
while len(true_set) < 500:
    text = random.choice(true_templates).format(
        univ=random.choice(univs), city=random.choice(cities), media=random.choice(media),
        percent=random.randint(60, 98)
    )
    if text not in true_set:
        true_set.add(text)
        all_data.append({"text": text, "label": 0})

# Shuffling the dataset to thoroughly mix true and fake records
random.shuffle(all_data)

# Save to target CSV with full UTF-8 configuration
df = pd.DataFrame(all_data)
df.to_csv('ETH_FAKE.csv', index=False, encoding='utf-8')

print("--- Data Generation Complete ---")
print(f"Total Rows Saved: {len(df)}")
print(f"Fake News (Label 1): {len(df[df['label'] == 1])} rows")
print(f"True News (Label 0): {len(df[df['label'] == 0])} rows")
print("\n--- Verification Check For Instructor ---")
print(df['label'].value_counts())