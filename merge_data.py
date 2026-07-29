import pandas as pd

# 1. Load your original English data
try:
    my_data = pd.read_csv('news_data.csv')
    print("Loaded news_data.csv")
except Exception as e:
    my_data = None
    print(f"Error loading news_data.csv: {e}")

# 2. Load the new Ethiopian dataset
try:
    eth_fake = pd.read_csv('ETH_FAKE.csv')
    print("Loaded ETH_FAKE.csv")
except Exception as e:
    eth_fake = None
    print(f"Error loading ETH_FAKE.csv: {e}")

# 3. Only combine if BOTH files loaded successfully
if my_data is not None and eth_fake is not None:
    combined_df = pd.concat([my_data, eth_fake], ignore_index=True)
    
    # 4. Save the combined file
    combined_df.to_csv('master_news_data.csv', index=False, encoding='utf-8-sig')
    print("--- Success! ---")
    print("New file 'master_news_data.csv' created.")
else:
    print("--- Failed ---")
    print("Could not create master file because one of the datasets is missing.")