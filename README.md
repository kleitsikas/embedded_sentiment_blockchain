# Embedded Sentiment in Blockchain Transactional Data

This is the code that accompanies the IEEE ICBC 2025 submission. 

To replicate the results of our study regarding the price prediction problem, you have to run the price_analysis/model_prediction_pipeline/ml_pipeline.ipynb jupyter notebook. The notebook contains markdowns with explanations on how to obtain our results as presented in Table V of the Results Section of the paper. 

To obtain the topic modelling results, you must refer to the topic_modelling folder which has the images of the intertopic distance maps as well as the python script to obtain both the figures and the Table IV of the Results Section, which contain the most salient topics of both LDA and BERTopic methods. 

The rest of the python files and datasets contain the whole process described in the methodology section of the paper. 

The originally fetched blockchain decoded text files (output_btc.csv and output_eth.csv) are too large to be pushed in Github (same applies for the eth_corpus.csv), but can be provided upon request. 