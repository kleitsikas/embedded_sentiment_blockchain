'''
'2018-04-24 19:03:29', 'All I can think of is bogdanoff saying PumpIt. Anyhow, pay no attention to the price. If you\'re a techie, you should focus on building new blockchain applications.'
Happy Birthday.... He sold? Pump It.',2022-01-05 06:14:37
EW BTC Pumpin',2015-07-20 23:26:38
EW ETH pumping, BTC dumping. Should I buy ETH, BTC or hold fiat?'",2016-01-23 12:17:42
EW If mike succeed i am going to dump',2015-08-19 07:43:42
'EW AshleyMadison leak dump: 40ae8a90de40ca3afa763c8edb43fc1fc47d75f1',2015-08-24 21:33:13
'EW DanielRWho1 #StraightSlime! #Spam #Burr #dump.coin',2016-12-27 21:01:09
EW Don't sell your bitcoins, it will be the world reserve so hold on to it!""",1000,2015-11-03 18:57:38
M.F. Holding from 10/2018 - better future, better BTC'",53327434,2018-12-09 21:08:56
EW We are reaching the Hearn threshold!',1000,2016-02-17 15:31:13
Be smarter be holder. We love you Eric.',9998029,2021-01-26 01:48:31
Be smarter be holder, I love you Tina.'",159996512,2021-01-28 12:01:07
BTC holds $ 647 awaiting Halving #Crypto',229626,2016-07-10 02:43:22
2: It's high, but I'm holdin' on. 4 letters""",360000,2019-02-11 20:44:48
HODL!!',12483236,2020-02-27 12:48:07
hodl on comrade!',10954713,2020-03-04 02:23:19
As elei\xc3\xa7\xc3\xb5es brasileiras foram fraudadas pelo TSE - venham me censurar! #HODL',19769,2022-11-05 05:16:04
I AM HODLING - GameKyuubi on 18th December 2013',5397,2022-12-19 23:15:13
audaces fortuna iuvat"" - HODL BTC - @machadop2p'",28700,2022-05-19 19:38:22
Hodl for yourself, but also for the ones who have  no alternative. WE ARE LEGION'",547467,2021-06-22 04:43:02
HODL a subset of 21 Mio."" clarifyBitcoin'",112694,2023-02-05 00:02:14
Sov Hodlr 13/Nov/2022 GET YOUR BITCOIN OFF THE EXCHANGE NOW! -Hodl Tarantula',7862,2022-11-13 23:12:47
Ramon S2 Larissa. Hodl forever 2021',1580862,2021-10-07 01:16:42
Hodling is using',1468543,2021-10-26 20:37:22
HODL!!!',1000,2017-09-13 11:57:50
Euk Sell Hang Seng HONK KONG Start a new short trade',1483241,2016-06-26 13:39:55
Euk Open Short GAS NATURAL SPAIN',1833241,2016-06-20 20:15:07
Sell FTSE Mib ITALY Start a new short trade-Net PositionShort 29.09 Cash 38.18',60000,2016-05-02 08:41:54
Open Long BMW Open Short SIEMENS Euk',2895241,2016-05-13 00:29:29
Closed short danone gain +2.48% open reverse long Euk',3095241,2016-05-03 22:41:09
Euk Close Short DANONE FRANCE Gain + 2.92%',2063241,2016-06-14 21:03:01
EW Fbi (annuiscoepit) best scam trader. Short bank's stocks, u will be rich""",1000,2016-06-12 11:33:36
EW #brexit short #Euro! long #bitcoin !',1000,2016-06-24 05:17:26
Sell Unilever UK Start a new short trade',983241,2016-07-04 09:26:44
Euk Close Short NESTLE' SWITZERLAND Gain + 1.71%""",2233241,2016-06-09 23:05:57
Euk Open Long ADIDAS GERMANY',2373241,2016-06-06 18:34:14
EW Fuck it, no patience. Max margin long bitcoin again.'",561000,2016-08-19 21:48:23
Saturday, March/05/2016 - Signal for today is 100 % BTC / 0 % USD'",3760000,2016-03-05 19:32:47
Saturday, March/19/2016 - Signal for today is 25 % BTC / 75 % USD'",3620000,2016-03-19 19:32:26
Signal for today is 0 % BTC / 100 % USD',87419,2015-08-09 19:41:18
Monday, October/19/2015- Signal for today is 100% BTC / 0% USD :Smart2'",190000,2015-10-19 20:01:34
    Signal for today is 25 % BTC / 75 % USD'",4600000,2015-12-12 20:16:34
# AROUND 395 SIGNALS!

btc,ce408ce7c8a11585bea08349cb0497c04cc893033e06049c3bdc420992d6e8e3,b'The signal gets lost in noise as easily as the SOUL gets lost without its MATE',42191,2023-04-28 16:25:45,0,op_return
'''
import pandas as pd
'''
df_btc = pd.read_csv('output_btc.csv',
                    delimiter=',',
                    header=0,
                    usecols=[2,4])


df_btc = pd.read_csv('output_eth.csv',
                    delimiter=',',
                    header=0)
'''
df_btc = pd.read_csv('price_data/HitBTC_BTCUSD_1h.csv',
                    delimiter=',',
                    header=0)

#df_btc['block_timestamp'] = pd.to_datetime(df_btc['block_timestamp'])
#df_btc = df_btc.sort_values(by='block_timestamp')
#df_btc['data'] = df_btc['data'].str.lower()
#df_btc = df_btc[df_btc['data'].str.contains('signal for today', na=False)]
#df_btc.to_csv('signals.csv', index=False)

#print(df_btc.head(5))
#print(df_btc.iloc[-1])
#print(len(df_btc))
pd.set_option('display.max_columns', None)
#print(df_btc.sample(10, random_state=43))
print(df_btc.head(6))