import pandas as pd

# 1. Load Marta's Work (Ethiopian & Local Data)
marta_amh = pd.read_csv('ETH_FAKE.csv', encoding='utf-8-sig')
marta_eng = pd.read_csv('news_data.csv')

# 2. Load Global Data (The new files in image_75807a.png)
# Using engine='python' prevents the crash your teammate had in Colab
isot_true = pd.read_csv('True.csv', engine='python')
isot_fake = pd.read_csv('Fake.csv', engine='python')

# Label the new data
isot_true['label'] = 'REAL'
isot_fake['label'] = 'FAKE'

# 3. Combine everything into Group 4 Master Dataset
# We align the columns to just 'text' and 'label'
master_df = pd.concat([
    marta_amh[['text', 'label']], 
    marta_eng[['text', 'label']], 
    isot_true[['text', 'label']], 
    isot_fake[['text', 'label']]
], ignore_index=True)

# 4. Export the final dataset
master_df.to_csv('master_news_data.csv', index=False, encoding='utf-8-sig')
print(f"--- SUCCESS: {len(master_df)} rows merged into master_news_data.csv ---")