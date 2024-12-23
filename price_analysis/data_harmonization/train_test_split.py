import pandas as pd

df = pd.read_csv('combined_daily_final_1.csv',
                    delimiter=',',
                    header=0)


split = int(0.85*len(df))

train_df = df[:split]
test_df = df[split:]

train_df.set_index('date', inplace=True)
train_df.to_csv(f'train_val_test/combined_train_final_1.csv', index=True)

test_df.set_index('date', inplace=True)
test_df.to_csv(f'train_val_test/combined_test_final_1.csv', index=True)
