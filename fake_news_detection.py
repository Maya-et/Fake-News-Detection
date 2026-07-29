import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

print("🔄 Loading dataset and training locked models...")
df = pd.read_csv('ETH_FAKE.csv', encoding='utf-8', on_bad_lines='skip')
df.columns = df.columns.str.strip()

text_col, label_col = df.columns[0], df.columns[1]
df = df.dropna(subset=[text_col, label_col])

# Shuffle once using a set random seed so it's always predictable
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

def preprocess_amharic(text):
    if not isinstance(text, str): return ""
    # Retain characters and space, stripping out noise symbols
    text = re.sub(r'[^\u1200-\u137F\s]', ' ', text)
    # Standardize common homophones to fix variations in spelling
    text = text.replace('ሐ', 'ሀ').replace('ኀ', 'ሀ').replace('ኻ', 'ሀ').replace('ሠ', 'ሰ').replace('ዐ', 'አ').replace('ፅ', 'ጸ')
    return re.sub(r'\s+', ' ', text).strip()

df['cleaned_text'] = df[text_col].apply(preprocess_amharic)
df = df[df['cleaned_text'].str.strip() != ""]

# Use explicit binary labels: 0 for True, 1 for Fake
df[label_col] = df[label_col].astype(str).str.strip()

# =========================================================================
# 🛠️ THE FIX: Upgraded Vectorizer for Context Tracking
# =========================================================================
# 1. Custom token pattern optimized for Ethiopic script space boundaries (\s)
# 2. ngram_range=(1, 3) captures groups of 2 and 3 words together (e.g., "አንበሶችን በሞተር ሳይክል")
# 3. sublinear_tf=True scales down word frequency weights so common words don't dominate
tfidf_vectorizer = TfidfVectorizer(
    token_pattern=r'[\u1200-\u137F]+', 
    ngram_range=(1, 3),
    sublinear_tf=True,
    max_features=10000
)
tfidf_train = tfidf_vectorizer.fit_transform(df['cleaned_text'])

# PassiveAggressiveClassifier works great for text, adding early stopping to prevent over-memorizing
pac = PassiveAggressiveClassifier(max_iter=100, random_state=7, early_stopping=True, validation_fraction=0.1)
pac.fit(tfidf_train, df[label_col])

# Quick local validation test print
train_preds = pac.predict(tfidf_train)
print(f"📈 Training Baseline Accuracy: {accuracy_score(df[label_col], train_preds) * 100:.2f}%")

# ==========================================
# 💾 EXPORT TRAINED OBJECTS TO DISK
# ==========================================
print("💾 Exporting production artifacts (`vectorizer.pkl` & `model.pkl`)...")
with open('vectorizer.pkl', 'wb') as v_file:
    pickle.dump(tfidf_vectorizer, v_file)

with open('model.pkl', 'wb') as m_file:
    pickle.dump(pac, m_file)

print("✅ Training complete! Brain files locked and ready for the app.")