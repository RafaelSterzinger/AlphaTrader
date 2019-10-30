# Project Proposal
## Scientific Papers
For my project in Applied Deep Learning I chose to focus on Deep Reinforcement Learning (DRL) in the financial market or rather on foreign exchange (forex) rates. It seems like this is a very hot topic, since there are dozens of scientific papers on sites like e.g [arXiv.org](https://arxiv.org/) covering this problem. Therefore, there are many possibilities in which this project might develop, but for the beginning I will use the following two papers:

* [Deep Reinforcement Learning in Financial Markets](https://arxiv.org/abs/1907.04373)

    Modelling the market as Markov Decision Process (MDP) and using DRL to trade.

* [Deep Reinforcement Learning for Foreign Exchange Trading](https://arxiv.org/abs/1908.08036)

    Comparing Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) algorithms and their effectiveness in the forex market. There are also possibilities to try different PPO algorithms from [this](https://arxiv.org/abs/1707.06347) paper.

----
There is also a whole different approach to this problem using images and Convolutional Neural Networks (CNN) or RNN's, with a focus on Long Short-Term Memory (LSTM). One of both methods may find application in the future, when the focus is on fine-tuning and varying the existing model.

**CNN's**

[Predict Forex Trend via Convolutional Neural Networks](https://arxiv.org/abs/1801.03018), [Conditional time series forecasting with convolutional neural networks](https://arxiv.org/abs/1703.04691), [Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market](https://arxiv.org/abs/1903.12258)

**RNN's**

[Stock Prices Prediction using Deep Learning Models](https://arxiv.org/abs/1909.12227), [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948), [Financial series prediction using Attention LSTM](https://arxiv.org/abs/1902.10877)

----
Depending on the speed of progress there is also the idea of including sentiment analysis in the model. Papers on this topic are: 

* [Forex trading and Twitter: Spam, bots, and reputation manipulation](https://arxiv.org/abs/1804.02233) => General research on the influence of Tweets on the market and whether to buy, hold or sell.

* [Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction](https://arxiv.org/pdf/1712.02136) => Using Hybrid Attention Networks (HAN), Recurrent Neural Networks (RNN) and a self-paced learning (SPL) mechanism to process recent news, related to the stock market.
----
[This](https://arxiv.org/abs/1910.05137) paper even goes one step further and tries to simulate the "whole stock market" in a multi agent system (MAS), where each agent learns individually and trades on its own. The collective behaviour of the agents is then used to predict the market. This method might be out of the projects scope at the moment due to missing processing power and time, but may be of interest in future work.

__Other__

[AlphaStock: A Buying-Winners-and-Selling-Losers Investment Strategy using Interpretable Deep Reinforcement Attention Networks](https://arxiv.org/abs/1908.02646)

## Topic
As already mentioned, this project will have a focus on __Reinforcement Learning (RL)__, especially in the context of forex trading and the prediction of this market.

## Project Type
Concerning the project type, there are many options applicable. Types like **Bring your own data, Bring your own method** and **Beat the stars** can all be applied, since the project can evolve in many directions. E.g. **Bring your own data** might be necessary if I focus on including sentiment analysis (Tweets) in the prediction, **Beat the stars** may be possible as well, since most of the selected scientific papers came out recently. 
If the project goes beyond the scope of this lecture I will solely focus on DRL and alter different approaches with RNN's or CNN's, which will at least result in **Bring your own method**.

## Summary
* __Description and Approach__

    The goal of the project is to predict different forex pairs like e.g. EUR-USD, USD-JPY, GBP-USD.

    Since I am rather new to machine learning, I will begin with standard DRL approaches listed on [SpinningUp](https://spinningup.openai.com/en/latest/user/algorithms.html) and their [Baseline Implementation](https://github.com/openai/baselines) to get an overview and a general practical understanding of this field as well as an insight on [Keras](https://keras.io/) or [PyTorch](https://pytorch.org/). Then I will have to chose one of the already mentioned more recent published scientific methods to predict the market with DRL. Knowledge I gathered till then will hopefully ease my decision.

    The chosen approach will then function as a baseline test for different variations and their performance. Variations I consider at the moment are the usage of RNN's, especially LSTM and/or news information in the form of Tweets. After my first tests, more specific rules for comparison will be set.
    
    For general comparison I will use a third party extension of the [OpenAI Gym Toolkit](https://github.com/openai/gym) called [AnyTrading](https://github.com/AminHP/gym-anytrading) - A testing and training environment to compare trading approaches.

* __Dataset__

    To gather the necessary data, I will use the [GoogleFinance API](https://support.google.com/docs/answer/3093281) with a corresponding [Python module](https://pypi.org/project/googlefinance.get/). Depending on how I will split the data into training and testing and if absolute prices, or just relative price changes will be used, there might be the necessity of preprocessing.

* __Work-Breakdown Structure__
__Work Description__

     (1) research topic and first draft
     
     First I focused my research on DRL with the idea of implementing an engine which masters the game Onitama, or any of the Atari Games provided by the OpenAI Gym. Since Onitama seemed to go beyond the lectures scope and the Atari-Games seemed to simple, I chose the topic of forex trading, which would also use DRL. Luckily there was already a training and testing environment, called AnyTrading, provided by OpenAI, which meant that I would have more time focusing on applying different methods instead of implementing a testing and training suit. After I chose my project I focused on getting an overview of different solutions to this problem by reading parts of the scientific papers as well as watching some online lectures from MIT. Lastly I wrote this description and started playing around with the use technologies which will be used.
     
     (2) setting up the environment and acquiring the datasets
     
     
     (3) designing and building an appropriate network
     
     
     (4) fine-tuning and varying that network
     
     
     (5) building an application to present the results
     

     (6) writing the final report
     
     
     (7) preparing the presentation of the project 
     
     
__Overview__
     
| Individual Task                                            | Time estimate        | Time used |
|------------------------------------------------------------|----------------------|-----------|
| research topic and first draft                             | 5h                   | 13h       |
| setting up the environment and acquiring the datasets      | 3h                   |           |
| designing and building an appropriate network              | 15h                  |           |
| fine-tuning and varying that network                       | 18h                  |           |
| building an application to present the results             | 6h                   |           |
| writing the final report                                   | 5h                   |           |
| preparing the presentation of the project                  | 3h                   |           |

