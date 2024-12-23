import pandas as pd
import spacy
#import nltk
import re
import gensim.downloader as gen
#import numpy
#import string
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    vader_sentiment = sia.polarity_scores(text)
    return vader_sentiment


def replace_abbreviations(text):
    abbreviations = {
        r'\bfomo\b': 'fear of missing out',
        r'\bbtd\b': 'buy the dip',
        r'\bshitcoin\b': 'shit coin',
        r'\bmemecoin\b': 'meme coin',
        r'\bieo\b': 'initial exchange offering',
        r'\bido\b': 'initial decentralised offering',
        r'\bico\b': 'initial coin offering',
        r'\btvl\b': 'total value locked',
        r'\bhodl\b': 'hold'
    }

    # Iterate over the dictionary and replace each abbreviation
    for abbr, full_form in abbreviations.items():
        text = re.sub(abbr, full_form, text)

    return text


def contains_term(text, terms_set):
    # Split the text into words and iterate through them
    for word in text.split():
        if word in terms_set:
            return True
    return False


def preprocess(text, excluded_tokens, three_letter_tokens):
    excluded_words = excluded_tokens
    three_letter_words = three_letter_tokens
    doc = nlp(text)
    filtered_tokens = []
    for token in doc:
        # commented out stuff in this part is to experiment with more raw text
        
        #if token.is_stop or token.is_punct or token.like_num:  
        #    continue
        
        #if len(token.text)  < 4 and token.text != 'buy':
        #    continue
        
        if len(token.text) > 30:
            continue

        #if token.is_oov and token.text not in excluded_words:
        #    continue

        #if token.text in excluded_words:
        #    filtered_tokens.append(str(token.text))
        #else:
        filtered_tokens.append(str(token.text)) # token.lemma_
    #print(filtered_tokens)
    doc = " ".join(filtered_tokens)
    if len(doc) < 4 and str(doc) not in three_letter_words:
        return " ", []
    else:
        return doc, filtered_tokens



crypto_finance_terms = ['ether', 'fiat', 'stable','money', 'funds', 'wealth', 'capital', 'euro', 'dollar', 'yen', 'pound', 'franc', 'rupee', 'ruble', 'yuan', 'won', 'lira', 'gold', 'silver', 'platinum', 'usd', 'eur', 'jpy', 'gbp', 'chf', 'inr', 'rub', 'cny', 'krw', 'try', 'aud', 'cad', 'sgd', 'nzd', 'mxn', 'brl', 'zar', 'hkd', 'sek', 'nok', 'finance', 'investment', 'asset', 'portfolio', 'equity', 'debt', 'credit', 'dividend', 'yield', 'liquidity', 'volatility', 'risk', 'return', 'bullish', 'bearish', 'trade', 'transaction', 'market', 'economy', 'hedge', 'inflation', 'deflation', 'recession', 'growth', 'valuation', 'margin', 'leverage', 'lever', 'broker', 'exchange', 'futures', 'options', 'bonds', 'stocks', 'shares', 'index', 'commodity', 'forex', 'cryptocurrency', 'crypto', 'bitcoin', 'ethereum', 'ripple', 'litecoin', 'blockchain', 'wallet', 'mining', 'hashrate', 'shitcoin', 'shit', 'altcoin', 'token', 'ico', 'smart', 'contract', 'nft', 'satoshi', 'digital', 'currency', 'virtual', 'ledger', 'public', 'key', 'private', 'decentralization', 'node', 'fork', 'gas', 'whitepaper', 'airdrop', 'hodl', 'fomo', 'fud', 'pump', 'dump', 'to the moon', 'mooning', 'buy', 'sell', 'long', 'short', 'bid', 'ask', 'spread', 'order', 'position', 'stop', 'loss', 'take', 'profit', 'bull', 'bear', 'rally', 'correction', 'trend', 'resistance', 'support', 'candlestick', 'chart', 'technical', 'analysis', 'fundamental', 'indicator', 'volume', 'breakout', 'macd', 'rsi', 'bollinger', 'bands', 'scheme', 'manipulation', 'insider', 'tip', 'inflate', 'plunge', 'artificial', 'hype', 'shill', 'fraud', 'scam', 'ponzi', 'pyramid', 'btc', 'eth', 'xrp', 'ltc', 'bch', 'ada', 'dot', 'link', 'bnb', 'xlm', 'uni', 'sol', 'trx', 'eos', 'xmr', 'xtz', 'wbtc', 'neo', 'vet', 'fil', 'dash', 'theta', 'etc', 'mkr', 'comp', 'zec', 'snx', 'dai', 'ksm', 'cel', 'yfi', 'uma', 'lend', 'bat', 'qtum', 'zrx', 'rvn', 'ht', 'pax', 'ont', 'nano', 'enj', 'dgb', 'signal', 'today', 'tomorrow','overbought', 'oversold', 'divergence', 'breakdown', 'break', 'reversal', 'spike', 'golden', 'cross', 'death', 'price', 'target', 'hit', 'moving', 'average', 'crossover', 'momentum', 'shift', 'accumulation', 'phase', 'distribution', 'pivot', 'point', 'fibonacci', 'retracement', 'line', 'channel', 'double', 'top', 'bottom', 'head', 'shoulders', 'inverse', 'cup', 'handle', 'pullback', 'capitulation', 'bounce', 'rebound', 'trap', 'squeeze', 'taking', 'building', 'scaling', 'in', 'out', 'allocation', 'diversification', 'gain', 'interest', 'rate', 'gdp', 'stock', 'ipo', 'fund', 'mutual', 'etf', 'bond', 'derivative', 'selling', 'trading', 'arbitrage', 'day', 'swing', 'scalping', 'cap', 'pe', 'ratio', 'earnings', 'report', 'management', 'class', 'securities', 'blue', 'chip', 'penny', 'sentiment', 'fiscal', 'policy', 'monetary', 'quantitative', 'easing', 'hedging', 'rating', 'junk', 'maturity', 'coupon', 'curve', 'proof', 'work', 'stake', 'stablecoin', 'address', 'fee', 'block', 'reward', 'consensus', 'algorithm', 'lightning', 'network', 'dex', 'initial', 'coin', 'offering', 'security', 'sto', 'sale', 'whale', 'bagholder', 'shilling', 'hard', 'soft', 'cold', 'storage', 'hot', 'nakamoto', 'bip', 'halving', 'layer-2', 'solution', 'cross-chain', 'atomic', 'sidechain', 'metaverse', 'wars', 'flash', 'loan', 'farming', 'pool', 'impermanent', 'rug', 'pull', 'tokenomics', 'dao', 'decentralized', 'autonomous', 'organization', 'speculation', 'capitalization', 'equities',  'income', 'sector', 'industry', 'commodities', 'derivatives', 'reach', 'swaps', 'bubble', 'crash', 'depression', 'surge','economic', 'cycle', 'tightening', 'execution', 'liquidation', 'rebalance', 'issuer', 'default', 'bankruptcy', 'insolvency', 'share', 'secondary', 'plummet','book', 'underwriter', 'prospectus', 'regulation', 'compliance', 'audit', 'tax', 'tariff', 'war', 'sanctions', 'embargo', 'negotiation', 'deal', 'merger', 'acquisition', 'takeover', 'joint', 'venture', 'partnership', 'synergy', 'restructuring', 'divestiture', 'spin-off', 'split', 'reverse', 'due', 'diligence', 'appraisal', 'research', 'forecast', 'prediction', 'projection', 'modeling', 'strategy', 'tactic', 'plan', 'objective', 'goal', 'benchmark', 'standard', 'metric', 'kpi', 'performance', 'improvement', 'optimization', 'efficiency', 'productivity', 'expansion', 'scale', 'economies', 'scope', 'diseconomies', 'competition', 'rivalry', 'monopoly', 'oligopoly', 'cartel', 'collusion', 'regulator', 'authority', 'central', 'bank', 'treasury', 'ministry', 'law', 'legislation', 'act', 'bill', 'statute', 'rule', 'guideline', 'ethics', 'integrity', 'transparency', 'accountability', 'responsibility', 'governance', 'oversight', 'supervision', 'inspection', 'examination', 'review', 'assessment', 'evaluation', 'investigation', 'probe', 'inquiry', 'hearing', 'trial', 'lawsuit', 'litigation', 'arbitration', 'mediation', 'settlement', 'agreement', 'arrangement', 'understanding', 'pact', 'accord', 'treaty', 'convention', 'protocol', 'alliance', 'coalition', 'collaboration', 'cooperation', 'coordination', 'integration', 'buyout', 'strategic', 'consortium', 'syndicate', 'group', 'association', 'institution', 'entity', 'body', 'agency', 'department', 'division', 'unit', 'branch', 'office', 'bureau', 'section', 'team', 'committee', 'panel', 'board', 'council', 'commission', 'watchdog', 'ombudsman', 'auditor', 'inspector', 'examiner', 'investigator', 'analyst', 'consultant', 'advisor', 'counselor', 'professor', 'facilitator', 'mediator', 'arbitrator', 'negotiator', 'diplomat', 'envoy', 'emissary', 'representative', 'delegate', 'liaison', 'agent', 'intermediary', 'go-between', 'conduit', 'pipeline', 'bridge', 'connector', 'interface', 'system', 'structure', 'framework', 'mechanism', 'process', 'procedure', 'method', 'approach', 'technique', 'program', 'project', 'initiative', 'campaign', 'movement', 'drive', 'effort', 'push', 'crusade', 'cause', 'mission', 'quest', 'pursuit', 'aim', 'ambition', 'aspiration', 'dream', 'vision', 'hope', 'desire', 'wish', 'yearning', 'longing', 'craving', 'hunger', 'thirst', 'passion', 'enthusiasm', 'zeal', 'eagerness', 'excitement', 'curiosity', 'intrigue', 'fascination', 'attraction', 'appeal', 'allure', 'charm', 'magnetism', 'seduction', 'temptation', 'enticement', 'lure', 'bait', 'draw', 'force', 'power', 'influence', 'impact', 'effect', 'consequence', 'result', 'outcome', 'product', 'output', 'benefit', 'advantage', 'earning', 'revenue', 'proceeds', 'rent', 'royalty', 'bonus', 'incentive', 'perk', 'fringe', 'compensation', 'remuneration', 'salary', 'wage', 'pay', 'payment', 'reimbursement', 'refund', 'rebate', 'discount', 'concession', 'allowance', 'subsidy', 'grant', 'advance', 'funding', 'financing', 'resource', 'property', 'possession', 'holdings', 'fortune', 'treasure', 'riches', 'abundance', 'plenty', 'prosperity', 'success', 'achievement', 'accomplishment', 'attainment', 'triumph', 'victory', 'conquest', 'domination', 'mastery', 'control', 'command', 'prestige', 'status', 'reputation', 'fame', 'renown', 'glory', 'honor', 'respect', 'esteem', 'admiration', 'adoration', 'worship', 'reverence', 'veneration', 'awe', 'wonder', 'amazement', 'surprise', 'shock', 'astonishment', 'bewilderment', 'perplexity', 'confusion', 'doubt', 'uncertainty', 'ambiguity', 'complexity', 'complication', 'aml', 'anti', 'laundering', 'kyc', 'know', 'your', 'customer', 'sec', 'irs', 'internal', 'service', 'taxation', 'gains', 'legal', 'tender', 'ban', 'restriction', 'license', 'enforcement', 'penalty', 'fine', 'financial', 'conduct', 'sanction', 'limit', 'regulatory', 'cftc', 'action', 'task', 'fatf', 'gdpr', 'protection', 'privacy', 'investor', 'consumer', 'prevention', 'disclosure', 'stability', 'fsb', 'crimes', 'fincen', 'foreign', 'assets', 'ofac', 'classification', 'utility', 'government', 'intervention', 'custody', 'rules'] + ['binance', 'coinbase', 'kraken', 'bitstamp', 'bittrex', 'gemini', 'bitfinex', 'huobi', 'kucoin', 'bitmart', 'gate.io', 'crypto.com', 'upbit', 'okex', 'bithumb', 'bitflyer', 'hitbtc', 'cex.io', 'poloniex', 'bitso', 'liquid', 'bitmex', 'bybit', 'bitbank', 'zaif', 'acx', 'indodax', 'exmo', 'luno', 'cointiger', 'coingecko', 'pancakeswap', 'uniswap', 'sushiswap', 'balancer', 'curve', '1inch', 'kyber', 'network', 'quickswap', 'bakeryswap', 'pancakebunny', 'ape swap', 'beefy', 'cream', 'ellipsis', 'alpaca', 'goose', 'auto', 'beethovenx', 'bunny park', 'olympus', 'dao', 'pangolin', 'snowball', 'snowbank', 'swamp', 'spiritswap', 'trader joe', 'traderjoe', 'trident', 'cometh', 'siren', 'bzx', 'volmex', 'dollar protocol', 'governor dao', 'harvest', 'pickle', 'yearn', 'keeperdao', 'vampire', 'darkpool', 'zero.exchange', 'cover protocol', 'picklejar', 'joe swap', 'scream', 'wootrade', 'paribus', 'bepswap', 'bunicorn', 'triton', 'dmdex', 'ellipses', 'falconswap', 'terraswap', 'julswap', 'lydia finance', 'definer', 'sswapp', 'the swap', 'smartdex', 'fyooz', 'blocktix', 'joys', 'oxbull.tech', 'sacoin', '1world', 'stackbit', 'coinfirm', 'fastswap', 'streamer data', 'equilibrium', 'belacoin', 'abs', 'deepbrain chain', 'qbic', 'jingtum tech', 'bitradio', 'obsidian', 'bitcoin cash', 'cardano', 'polkadot', 'chainlink', 'binance coin', 'stellar', 'tether', 'usd coin', 'uniswap', 'dogecoin', 'solana', 'eos', 'monero', 'tezos', 'wrapped bitcoin', 'neo', 'vechain', 'filecoin', 'theta', 'etc', 'maker', 'compound', 'zcash', 'synthetix', 'dai', 'avalanche', 'kusama', 'celsius', 'yearn.finance', 'uma', 'lend', 'basic attention token', 'qtum', 'ravencoin', 'huobi', 'paxos', 'ontology', 'nano', 'enjin', 'digibyte', 'fear of missing out', 'pumpit', 'dumpit', 'total value locked']
three_letter_tokens = ['ht', 'in', 'pe', 'acx', 'dao', 'bzx', 'abs', 'eos', 'neo', 'etc', 'dai', 'uma', 'yen', 'won', 'usd', 'eur', 'jpy', 'gbp', 'chf', 'inr', 'rub', 'cny', 'krw', 'try', 'aud', 'cad', 'sgd', 'nzd', 'mxn', 'brl', 'zar', 'hkd', 'sek', 'nok', 'ico', 'nft', 'key', 'gas', 'fud', 'buy', 'bid', 'ask', 'rsi', 'tip', 'btc', 'eth', 'xrp', 'ltc', 'bch', 'ada', 'dot', 'bnb', 'xlm', 'uni', 'sol', 'trx', 'eos', 'xmr', 'xtz', 'neo', 'vet', 'fil', 'etc', 'mkr', 'zec', 'snx', 'dai', 'ksm', 'cel', 'yfi', 'uma', 'bat', 'zrx', 'rvn', 'pax', 'ont', 'enj', 'dgb', 'hit', 'top', 'cup', 'out', 'gdp', 'ipo', 'etf', 'day', 'cap', 'fee', 'dex', 'sto', 'hot', 'bip', 'rug', 'dao', 'tax', 'war', 'due', 'kpi', 'law', 'act', 'aim', 'pay', 'awe', 'aml', 'kyc', 'sec', 'irs', 'ban', 'fsb']
three_letter_tokens_reduced =  ['ht', 'in', 'pe', 'acx', 'dao', 'bzx', 'abs', 'eos', 'neo', 'etc', 'dai', 'uma', 'yen', 'won', 'usd', 'eur', 'jpy', 'gbp', 'chf', 'inr', 'rub', 'cny', 'krw', 'aud', 'cad', 'sgd', 'nzd', 'mxn', 'brl', 'zar', 'hkd', 'sek', 'nok', 'key', 'gas', 'fud', 'bid', 'ask', 'rsi', 'tip', 'btc', 'eth', 'xrp', 'ltc', 'bch', 'ada', 'dot', 'bnb', 'xlm', 'uni', 'sol', 'trx', 'eos', 'xmr', 'xtz', 'neo', 'vet', 'fil', 'etc', 'mkr', 'zec', 'snx', 'dai', 'ksm', 'cel', 'yfi', 'uma', 'bat', 'zrx', 'rvn', 'pax', 'ont', 'enj', 'dgb', 'fee', 'dex', 'sto', 'hot', 'bip', 'rug', 'dao', 'awe', 'aml', 'fsb']
filtered_crypto_finance_terms = [x for x in crypto_finance_terms if x not in three_letter_tokens_reduced]
crypto_finance_terms = set(filtered_crypto_finance_terms)
three_letter_tokens = set(three_letter_tokens)

## only for the ethereum topic modelling so that to include all kinds of text ##
#crypto_finance_terms = []
#three_letter_tokens = []
##########

print("crypto/finance words: ",len(crypto_finance_terms))
nlp = spacy.load("en_core_web_lg")
ar = nlp.get_pipe('attribute_ruler')
ar.add([[{"TEXT":"hodl"}]], {"LEMMA": "hold"})

#doc = nlp("crypto pump dump btc eth sell buy hodl hold cryptocurrency blockchain bitcoin ethereum shitcoin altcoin dogecoin aml ath bearish bear market bollinger btd bullish bull deadcoin fomo hedging ico ido ieo memecoin moon roi scalping ")
#for token in doc:
#    if token.is_oov or token.has_vector == False: 
#        print(token.text, "Vector", token.has_vector, "OOV", token.is_oov, "LEMMA :", token.lemma_) 

#wv = gen.load("word2vec-google-news-300")
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=128, truncation=True, padding = 'max_length')

coin = 'btc' #eth
df = pd.read_csv(f'text_data/output_{coin}.csv')
df_tada = pd.DataFrame(columns=['text', 'cryptobert_sentiment', 'subjectivity', 'polarity', 'compound', 'neg', 'neu' ,'pos' , 'block_timestamp', 'tx_type', 'value'])

df_tada = pd.DataFrame(columns=['text', 'block_timestamp'])
df = df.sort_values(by='block_timestamp')
df.drop(['chain','hash','value','to_contract','type'], inplace=True, axis=1)
#####
#yes, for loop could have been avoided with pandas operations for efficiency
for i in tqdm(range(len(df.index))):
    if i not in df.index:
        continue
    tmstp = pd.to_datetime(df.loc[i,"block_timestamp"])
    #tx_type = str(df.loc[i,"type"])
    #value = str(df.loc[i,"value"])
    doc = str(df.loc[i,"data"])
    doc = doc[2:-1]
    #remove frequent ads
    
    # commented out for eth all blockchain topic modeling
    #if doc == 'BFX_REFILL_SWEEP' or doc == 'hotwallet drain fee' or doc == 'Bitzlato' or doc == '503: Bitcoin over capacity!' or doc == 'Bitcoin: A Peer-to-Peer Electronic Cash System' or doc == 'We\'ll buy your Bitcoins. sell.buy.bitcoin@protonmail.com' or doc == ' WWW.BTCKEY.ORG  Bitcoin wallet recovery and wallet decryption' or doc == ' BTCKEY.ORG  Bitcoin wallet decryption and private key recovery' or doc == ' WWW.BTCKEY.CO  Buy your empty private key':
    #    continue
    
    #Do not include base64 encoded pattern (usually ends with '==')
    if re.fullmatch(r'[A-Za-z0-9+/]*={0,2}', doc) and len(doc) % 4 == 0:
        continue
    # Remove links
    doc = re.sub(r'https?://\S+', '', doc)
    

    # Remove symbols (punctuation and special characters)
    #doc = re.sub(r'[^\w\s]', ' ', doc)
    
    
    # Remove numbers between letters
    #doc = re.sub(r'(?<=\D)\d+(?=\D)', '', doc)
    
    
    #Do not include text that has no white spaces and is longer than 25 characters (noice)
    if (' ' not in doc and len(doc) > 25):
        continue
    #Do not include text that only contains 2 letters
    if len(doc) < 3:
        continue
    #Do not include text that only contains numbers
    if doc.isdigit():
        continue

    #doc = doc.lower()
    
    doc = replace_abbreviations(doc)
    doc, filtered_tokens = preprocess(doc, crypto_finance_terms, three_letter_tokens)
    
    #remove anything that is not financially related
    #
    #result = contains_term(doc, crypto_finance_terms)
    #if result == False:
    #    continue
    #

    # no context in one word after keeping
    words = doc.split()
    if len(words) < 2:
        continue

    ''' 
    #we get the sentiment in harmonization.py, once the texts are aggregated by day
    subjectivity = getSubjectivity(doc)
    polarity = getPolarity(doc)
    SIA = getSIA(doc)
    compound = SIA['compound']
    neg = SIA['neg'] 
    neu = SIA['neu']
    pos = SIA['pos']
    sent_doc = [doc]
    sentiment = pipe(sent_doc)
    if sentiment[0]['label'] == 'Bullish':
        label = 2
    if sentiment[0]['label'] == 'Neutral':
        label = 1
    if sentiment[0]['label'] == 'Bearish':
        label = 0
    '''
    if doc == " ":
        continue
    #sentence_vector = wv.get_mean_vector(filtered_tokens, pre_normalize=True) #includes normalization by default
        
    #commented out for eth topic modelling
    #df_tada = df_tada._append({'text': doc, 'cryptobert_sentiment': label, 'subjectivity': subjectivity, 'polarity': polarity, 'compound': compound , 'neg': neg, 'neu': neu, 'pos': pos, 'block_timestamp': tmstp, 'tx_type': tx_type, 'value': value}, ignore_index=True)
    df_tada = df_tada._append({'text': doc, 'block_timestamp': tmstp}, ignore_index=True)
df_sorted = df_tada.sort_values(by='block_timestamp')
df_sorted.to_csv(f'{coin}_corpus_raw.csv', index=False)