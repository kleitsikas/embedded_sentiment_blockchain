import pandas as pd
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import sys
sys.set_int_max_str_digits(500000)
from decimal import Decimal
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    vader_sentiment = sia.polarity_scores(text)
    return vader_sentiment

def concat_text(array):
    texts = list(array)
    
    #processed_texts = []
    
    #unique_texts = list(set(array)) #change to preserve order
    
    #for text in texts:
    #    cleaned_text = remove_duplicates(text) # remove error
    #    label, neu = feature_eng_beta(cleaned_text)
    #    if int(label) != 1 or int(neu) != 1:
    #        processed_texts.append(cleaned_text)
    
    #print(unique_texts)
    #aggregated_texts = " ".join(processed_texts)
    aggregated_texts = " , ".join(texts)
    #keep each word only once
    #aggregated_texts = remove_duplicates(aggregated_texts)
    return aggregated_texts

def remove_duplicates(text):
    words = text.split()
    unique_words = []
    for word in words:
        if word not in unique_words:
            unique_words.append(word)
    return ' '.join(unique_words)

'''
def concat_text(array):
    # Initialize an empty list to store decoded texts
    decoded_texts = []
    
    # Iterate over the array, convert each integer to bytes, and then decode
    for number in array.values:
        number = Decimal(number)
        number = int(number)
        # Calculate the number of bytes needed for each integer
        num_bytes = (number.bit_length() + 7) // 8
        # Convert the integer to bytes and then decode
        text = number.to_bytes(num_bytes, 'little').decode('utf-8')
        decoded_texts.append(text)
    
    # Create a set of unique texts and join them
    unique_texts = set(decoded_texts)
    unique_texts = text_to_byte(" ".join(unique_texts))
    return unique_texts
'''
def feature_eng(row):
    print(row)
    doc = str(row)
    subjectivity = getSubjectivity(doc)
    polarity = getPolarity(doc)
    SIA = getSIA(doc)
    compound = SIA['compound']
    neg = SIA['neg'] 
    neu = SIA['neu']
    pos = SIA['pos']
    sent_doc = [doc]
    sentiment = pipe(sent_doc)
    sentiment_2 = classifier_2(doc)
    if sentiment[0]['label'] == 'Bullish':
        label = 2
    if sentiment[0]['label'] == 'Neutral':
        label = 1
    if sentiment[0]['label'] == 'Bearish':
        label = 0
    if sentiment_2[0]['label'] == 'LABEL_0':
        label_2 = 0
    if sentiment_2[0]['label'] == 'LABEL_1':
        label_2 = 1   
    return label, label_2, subjectivity, polarity, compound, neg, neu, pos

def feature_eng_beta(row):
    #print(row)
    doc = str(row)
    sent_doc = [doc]
    sentiment = pipe(sent_doc)
    SIA = getSIA(doc)
    neu = SIA['neu']
    if sentiment[0]['label'] == 'Bullish':
        label = 2
    if sentiment[0]['label'] == 'Neutral':
        label = 1
    if sentiment[0]['label'] == 'Bearish':
        label = 0  
    return label, neu

def window(iterable, size, total_list):
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    total_list.append(win)
    for e in i:
        win = win[1:] + [e]
        total_list.append(win)
    return total_list


model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=512, truncation=True, padding = 'max_length')

model_name_2 = "kk08/CryptoBERT"
tokenizer_2 = BertTokenizer.from_pretrained(model_name_2)
model_2 = BertForSequenceClassification.from_pretrained("kk08/CryptoBERT")
classifier_2 = pipeline("sentiment-analysis", model=model_2, tokenizer=tokenizer_2)

#changed folder organization so path must change
df_btc = pd.read_csv('../btc_corpus_raw_financial.csv',
                    delimiter=',',
                    header=0,
                    usecols=[0,1]) #[0,8] for non-raw data


df_eth = pd.read_csv('../eth_corpus_raw_financial.csv', 
                    delimiter=',',
                    header=0,
                    usecols=[0,1])

df_btc['block_timestamp'] = pd.to_datetime(df_btc['block_timestamp'])
df_btc = df_btc.sort_values(by='block_timestamp', ascending=True)
df_eth['block_timestamp'] = pd.to_datetime(df_eth['block_timestamp'])
df_eth = df_eth.sort_values(by='block_timestamp', ascending=True)

# for hours
#start_date = '2017-10-25 18:00:00'  # 
#end_date = '2018-06-10 21:00:00'    # 

# for days
start_date = '2017-11-11'  # 
end_date = '2023-06-11'    # 


filtered_btc = df_btc[(df_btc['block_timestamp'] >= start_date) & (df_btc['block_timestamp'] <= end_date)]
filtered_eth = df_eth[(df_eth['block_timestamp'] >= start_date) & (df_eth['block_timestamp'] <= end_date)]

combined_df = pd.concat([filtered_btc, filtered_eth], ignore_index=True)
combined_df = combined_df.sort_values(by='block_timestamp', ascending=True)

#~~~~~~~ CODE FOR DAILY WITHOUGHT ROLLING WINDOW ~~~~~~~~#

combined_df['block_timestamp'] = pd.to_datetime(combined_df['block_timestamp'], format='%Y-%m-%d H:M:S')
combined_df = combined_df.sort_values(by='block_timestamp', ascending=True)

#combined_df['block_timestamp'] = combined_df['block_timestamp'].dt.strftime('%Y-%m-%d')
#combined_df['block_timestamp'] = pd.to_datetime(combined_df['block_timestamp'])

#combined_df = combined_df.sort_values(by='block_timestamp', ascending=True)
combined_df.set_index('block_timestamp', inplace=True)
print(len(combined_df))

combined_df = combined_df['text'].resample('D').apply(concat_text)
combined_df = combined_df.to_frame(name='text')
combined_df = pd.concat([combined_df, combined_df.apply(feature_eng, axis=1, result_type='expand')], axis='columns')
combined_df.columns = ['text', 'cryptobert_sentiment', 'cryptobert_sentiment_2', 'subjectivity', 'polarity', 'compound', 'neg', 'neu', 'pos'] 

coin = 'btc'
ts_btc = pd.read_csv(f'../price_data/{coin}_daily_combined_bloomberg_large.csv',
                        delimiter=',',
                        header=0,
                        index_col=0,
                        parse_dates=True,
                        date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d")
                        )

coin = 'eth'
ts_eth = pd.read_csv(f'../price_data/{coin}_daily_combined_bloomberg_large.csv',
                        delimiter=',',
                        header=0,
                        index_col=0,
                        parse_dates=True,
                        date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d")
                        )


combined_df = pd.concat([combined_df, ts_btc], axis=1)
combined_df = pd.concat([combined_df, ts_eth], axis=1)
combined_df.index.name = 'date'
combined_df.to_csv(f'combined_daily_final_raw.csv', index=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#combined_df = filtered_btc.sort_values(by='block_timestamp')
#print(len(combined_df))

#combined_df.to_csv('aaa.csv', index=True)
#window_size = 6  # hours
#final_df = custom_rolling_concat(combined_df, 'text', window_size)
#combined_df['text'] = combined_df['text'].apply(text_to_byte)
#final_df = combined_df['text'].resample('6H', label='right').apply(concat_text) #H for hour, D for day
#print(final_df.head(50))
#combined_df['text'] = combined_df['text'].fillna(0).apply(int)
#final_df = combined_df['text'].resample('6H', label='right').apply(concat_text) #H for hour, D for day
#final_df = custom_rolling_concat(combined_df, 'text', window_size='3H')
#final_df = combined_df['text'].rolling(window='3H', closed='left').apply(concat_text)
#final_df = pd.DataFrame(final_df)
#final_df = final_df[final_df.columns[0]].apply(decode_bytes)
#final_df = combined_df['text'].rolling(window='6H', closed='left').apply(concat_text)
'''
combined_df["block_timestamp"] = combined_df["block_timestamp"].dt.floor('H')
combined_df.index = combined_df["block_timestamp"]
combined_df.drop(['block_timestamp'], inplace=True, axis=1)
combined_df = combined_df['text'].resample('1H').apply(concat_text)
combined_df = combined_df.to_frame(name='text')


#combined_df.to_csv('aaa.csv', index=True)

tx_list = []
chunks_list = []
date_format_str = '%Y-%m-%d %H:%M:%S' # Given timestamp in string
n_hours = 1 #DEFINE GRAPH TIMESPAN

base_date = pd.to_datetime('2017-10-25 18:00:00')
#base_date = datetime.strptime(base_date, date_format_str)

for line in range(len(combined_df.index)):  #requires asceding order
    #print(base_date, combined_df.index[line])
    given_time = combined_df.index[line]
    if (given_time < (base_date+ pd.DateOffset(hours=n_hours))):
        #print(json.loads(line)['block_timestamp'])
        tx_list.append((combined_df['text'][line], combined_df.index[line]))
    else:
        chunks_list.append(tx_list)
        tx_list = []
        tx_list.append((combined_df['text'][line], combined_df.index[line])) #prwto stoixeio sthn epomenh lista, to prwto stoixeio sto opoio spaei o elegxos
        base_date += pd.DateOffset(hours=n_hours)

tx_list = []
total_list = []
size = 24 # define lists timespan
#print(chunks_list[:10])
window(chunks_list,size, total_list)  #total_list = [ [[],[],[],[],[]] , [[],...,[]] , [..[]..]  ]
#print(len(total_list))  
new_total_list = []
for i in total_list:
    new_total_list.append([x for l in i for x in l])   #new_total_list = [ [5h_list] , [5h_list] , [....]  ] 
                                                       #oi listes autes einai h mia +1h apo tin prohgoumenh
                                                       #logw sliding window (an theloume na pianoume tis akmes)
                                                       #den ginetai na glutwsoume tous ypologismous autous

print(new_total_list[:5])

processed_data = {}
for sublist in new_total_list:
    unique_texts = set(text for text, _ in sublist if text)  # Collecting unique non-empty texts
    concatenated_text = ' '.join(unique_texts).strip()
    last_timestamp = sublist[-1][1]
    processed_data[last_timestamp] = concatenated_text

# Create DataFrame
fina_df = pd.DataFrame(list(processed_data.items()), columns=['Date', 'Text'])
fina_df.set_index('Date', inplace=True)

final_df = pd.concat([fina_df, fina_df.apply(feature_eng, axis=1, result_type='expand')], axis='columns')
final_df.columns = ['text', 'cryptobert_sentiment', 'cryptobert_sentiment_2', 'subjectivity', 'polarity', 'compound', 'neg', 'neu', 'pos'] 

ts = pd.read_csv('../price_data/eth_usd_combined.csv',
                        delimiter=',',
                        header=0,
                        index_col=0,
                        parse_dates=True,
                        date_parser=lambda x: pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")
                        )


final_df = pd.concat([final_df, ts], axis=1)
final_df.index.name = 'date'
final_df.to_csv('not_mixed_final_dataset_eth_combined.csv', index=True)
'''
