In the financial sector, accurate stock price prediction is crucial for investors seeking to
maximize returns while minimizing risk. For this project, we employed Hidden Markov Models
to predict the future stock price of 10 top tech companies, given both the previous 30 days of
stock price as well as 25 news headlines from each day, to determine if these factors could
meaningfully predict future stock movements. The news headlines were analyzed with cosine
similarity and language models to detect whether the news was beneficial for the company, with
the sentiment scores from these headlines being passed into the stock prediction model.
We defined our state transition matrix with a linear regression model from scikit-learn to
predict today’s stock performance based on previous 30 days’ data. For our observation matrix,
we utilized the pretrained all-MiniLM-L6-v2 sentiment analysis model to evaluate each day’s
news headlines. We combined these two models to construct our Hidden Markov Model.
The linear regression model was trained on historical stock data for 10 top tech
companies from Yahoo Finance, covering the period from July 2008 to July 2014. The fine-tuned
model was then tested on data from July 2014 to July 2016 to predict stock prices. For sentiment
analysis, we applied the all-MiniLM-L6-v2 model to a news dataset from the Kaggle
Competition "Daily News for Stock Market Prediction" to obtain sentiment results for each day’s
top 25 news headlines.
The Linear Regression model served as a baseline for comparison with our integrated
HMM. The Linear Regression model only considered the stock’s previous data. After training,
the Linear Regression reached a high R^2 score, indicating that the predicted and true price
movements were similar overall (as shown in Figure 1). However, we believe the most important
thing in stock prediction is to see if the predicted stock trend would be the same as the true stock
trend. If the stock price prediction for tomorrow shows the stock price would go up, then we
would want to know if the stock price would really increase. Therefore, we implemented our
own evaluation metrics which calculated the percentage of successful stock trend prediction. We
tested the linear regression baseline model on our customized metrics. The directional accuracy
was low. For example, the linear regression model for Meta only got an accuracy of 46.51%,
meaning that the prediction was not precise on a day-to-day stock trend basis.
Figure 1: true prices vs predicted prices - Meta
For the sentiment analysis, we first calculated the cosine similarity between the target
company’s name and the news headlines. In many cases, the news articles may have not directly
implied a relation to a stock, so a similarity score was implemented to ensure that the news
headlines we choose are relevant to our target stock. If the absolute value of cosine similarity is
larger than 0.2, we will do sentiment analysis on that news headline. Our sentiment analysis
model all-MiniLM-L6-v2 model returns a list of sentiment prediction and corresponding
confidence score.
[{'label': 'positive', 'score': 0.9930960536003113}, {'label': 'neutral', 'score':
0.005177223589271307}, {'label': 'negative', 'score': 0.0017267182702198625}]
From the news headline dataset, for each day, there are 25 news headlines available. We average
out all sentiment analysis results each day to get a final sentiment analysis result. We stored the
sentiment data along with the stock prediction result and passed them into the Hidden Markov
Model.
We implemented the Hidden Markov Model as we learned in lecture Markov Chains. The
relationship between each state was described by the linear regression model, while the
relationship between state and observation was described by the sentiment analysis model. We
defined our transition matrix with the confidence score from sentiment analysis model: the
probability that the stock price would go below the predicted price was represented by the
confidence score for negative sentiment, the probability that the stock price would stay the same
as the predicted price was the confidence score for neutral sentiment, and the stock price would
go above the predicted price was represented by the confidence score for positive sentiment.
Shown in the table below, our Hidden Markov Model performed similarly to the baseline
linear regression model. Each model outperformed with 5 out of 10 companies in the evaluation.
Stock HMM Accuracy HMM MSE LR Accuracy LR MSE
Meta 0.4498 5.4199 0.4651 2.9811
Google 0.4968 1.3218 0.5031 0.2712
Microsoft 0.4968 1.1954 0.5052 0.6500
Apple 0.4925 0.3815 0.4608 0.2098
IBM 0.5351 6.3924 0.4947 4.0220
NVIDIA 0.4754 0.0005 0.4651 0.0002
Tesla 0.5053 0.2772 0.4693 0.1679
Salesforce 0.4691 3.6080 0.5243 2.0039
AMD 0.5394 0.0235 0.4630 0.0149
Amazon 0.4989 0.4483 0.5222 0.2713
Table 1: Hidden Markov Model vs Linear Regression stats
The reason that HMM did not outperform the LR model could be the news headlines
dataset being too sparse. We calculated how much news was correlated with our target stock. We
realized that only 3-4% news headlines in the dataset were related to the target company, based
on the results from cosine similarity. We expect the HMM performance to increase if we could
use a news dataset with more informative news headlines that include more data related to the
target company. The underperformance of the HMM compared to the LR model could also be
attributed to our design of the Hidden Markov Model. Our customized HMM potentially needed
some improvement. We customized the transition matrix as well as the observation matrix which
potentially could contain bugs. Furthermore, the state model we used may not be informative
enough, which could also contribute to the HMM's poor performance.
In conclusion, the HMM's performance was limited by the sparsity of the news headlines
dataset and potential issues in our customized model. Enhancing the dataset with more relevant
news and refining the HMM's design could lead to better predictive performance.
