# 
DEEPREVIEW - SENTIMENT ANALYSIS OF PRODUCT REVIEWS USING DEEP LEARNING



Abstract:
Sentiment analysis is one of the fastest growing research area, which helps customers to make better-informed purchase decisions through proper understanding and analysis of collective sentiments from the web and social media. It also provides organizations the ability to measure the impact of their social marketing strategies by identifying the public emotions towards the product or the events associated to them. Most of the studies done so far have focused on obtaining sentiment features by analyzing syntactic and lexical features that were explicitly expressed through sentiment words, emoticons and other special symbols. In this paper, we propose an approach to carry out the sentiment analysis of product reviews using deep learning. Unlike traditional machine learning methods, deep learning models do not depend on feature extractors as these features are learned directly during the training process. 
In this paper test for sentiment classification method with product reviews of mobile phones gathered from Amazon, IMDB and Yelp and show that our method gives better prediction accuracy than most of the existing method.



Dataset Description:
There are 3 datasets: Yelp, IMDB, Amazon these static datasets. The dataset is constructed with 1000 reviews with labels. The idea is to evaluate the performance of our proposed CNN architecture on these product review, each product review includes 500 positive and 500 negative reviews. To make the evaluation process more precise 5761 reviews are chosen from 4,00,000 reviews released by amazon, where customers give rating from 1 to 5 for their reviews, this project uses reviews rated as 5 rating for their reviews, this project uses reviews 1 (0-polarity) and reviews rated as 5 (1- polarity). The goal was for no neutral sentences to be selected.


Model:
Initially, the reviews and labels are extracted from the text file and different preprocessing steps like cleaning stopword , numerics, etc are performed using Neattext and Nltk library. The obtained sentence matrix , concatenated with the polarity (label of each review) is then fed as input to the CNN(convolutional neural network). CNN architecture in this work consists of Convolutional layer, having kernel size 8 and 300 feature maps(outputs of convolutional layer) followed by ReLU activation,  Max-pooling, Dropout,  Dense fully connected layer and Softmax activation function to get values in the range range(0 to 1).
The advanced second model is a product review classification model that predicts whether input text belongs to a certain category or not. It has four layers: an embedding layer, a bidirectional LSTM layer, a dense layer with ReLU activation, and a final dense layer with sigmoid activation. The embedding layer converts each word in the input text into a vector representation. The bidirectional LSTM layer processes the input text in both directions to understand its context and meaning. The model is compiled with a binary cross-entropy loss function and an accuracy metric to measure its performance.
Results:  
CNN Model:
Training  Accuracy: 97.14%
Training Loss: 0.0522
Test Accuracy: 48.8%
LSTM Model:
Training  Accuracy: 98.96%
Training Loss: 0.0309
Test Accuracy: 91.6%
       
Conclusion:
Neural networks and NLP are becoming popular in solving almost any machine learning classification problem. Our experiment results show that the proposed approach entitles a better accuracy compared to the existing traditional machine learning models and the accuracy of the model increases with the use of LSTM neural network.
