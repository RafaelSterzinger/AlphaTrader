# Project Proposal
## Scientific Papers
---
For my project in Applied Deep Learning I chose to focus on Deep Reinforcment Learning (DRL) in the financial market or rather on forex exchange market. It seems like this is a very hot topic, since there are douzens of scientific papers on sites like e.g [arXiv.org](https://arxiv.org/). Therefore there are many possibilities in which direction this project will develop itself. For the beginning, as I mentioned, I will have an emphasise on DRL. Papers I found are e.g.:

* [Deep Reinforcement Learning in Financial Markets](https://arxiv.org/abs/1907.04373)

    Modeling the market as Markov Decision Process (MDP) and using Deep RL to trade.

* [Deep Reinforcement Learning for Foreign Exchange Trading](https://arxiv.org/abs/1908.08036)

    Comparing DQN and PPO (Proximal Policy Optimization) algorithms and their effectivness in the forex market. Posibility to try different PPO algorithms from [this](https://arxiv.org/abs/1707.06347) paper.

Depending on the speed of progress there is also the idea of including senitment analysis in the model. Papers on this topic are: 

* [Forex trading and Twitter: Spam, bots, and reputation manipulation](https://arxiv.org/abs/1804.02233) => General research on the influence of Tweets on the market and whether to buy, hold or sell)

* [Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction](https://arxiv.org/pdf/1712.02136) => Using Hybrid Attention Networks (HAN), Recurrent Neural Networks (RNN) and a self-paced learning (SPL) mechanism to process recent news, related to the stock market.

There is a whole different approach to this problem using images and CNN's or LSTM, which is out of the projects scope at the moment.

**CNN's**

[Predict Forex Trend via Convolutional Neural Networks](https://arxiv.org/abs/1801.03018), [Conditional time series forecasting with convolutional neuralnetworks](https://arxiv.org/abs/1703.04691), [Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market](https://arxiv.org/abs/1903.12258)

**LSTM**

[Stock Prices Prediction using Deep Learning Models](https://arxiv.org/abs/1909.12227), [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948), [Financial series prediction using Attention LSTM](https://arxiv.org/abs/1902.10877)

[This](https://arxiv.org/abs/1910.05137) paper even goes one step further and tries to simulate the whole stock market in a multi agent system (MAS), where each agent learns individually and trades on its own. The collective behaviour of the agents is then used to predict the market. This might be out of scope as well, due to lacking proccesing power.

__Other__

[AlphaStock: A Buying-Winners-and-Selling-Losers Investment Strategy using Interpretable Deep Reinforcement Attention Networks](https://arxiv.org/abs/1908.02646)

## Topic
As already mentioned, the topic of this projects will have an emphasise on Reinforcement Learning (RL), especially in the context of forex exchange and the prediction of this market.
## Project Type
Concerning the project type, there are many options applicable. Types like **Bring your own data, Bring your own method and Beat the stars** can all be applied, since the project can evolve in many directions. E.g. **Bring your own data** might be necessary when I focus on including sentiment analysis (Tweets) in the prediction, **Beat the start** may be possible as well, since most of the selected scientific papers came out recently. If the progress of the project is not as fast as I have whished and because of it I will solely focus on DRL, then at least I can alter different approaches => **Bring your own method**
## Summary
* __Description and Approach__

    The goal of the project is to predict different forex pairs like e.g. EUR-USD, USD-JPY, GBP-USD.

    Since I am rather new to machine learning, I will begin with standard DRL approches listed on [SpinningUp](https://spinningup.openai.com/en/latest/user/algorithms.html) and their [Baseline Implementation](https://github.com/openai/baselines) to get an overview and a general practical understanding of this field. Then I will have to chose one of the more recent published scientific methods to predict the market which I mentioned early. Knowledge I gathered till then will hopefully ease my decision.

    The chosen approach will then funciton as a baseline test for different variations and their performance. Variations I might consider at the moment are the inclusion of RNN's, especially LSTM and news information in the form of Tweets. For the implementation I will use a third party extension of the [OpenAI Gym Toolkit](https://github.com/openai/gym) called [AnyTrading](https://github.com/AminHP/gym-anytrading) - A testing and training environment to compare trading approaches, approved by OpenAI.

* __Dataset__

    To gather the necessary data, I will use the [GoogleFinance API](https://support.google.com/docs/answer/3093281) with a corresponding [Python module](https://pypi.org/project/googlefinance.get/).

* __Work-Breakdown Structure__


c.A work-breakdown structure for the individual tasks with time estimates for dataset collection;

     1. designing and building an appropriate network;


     2. training and fine-tuning that network;


     3. building an application to present the results;


     4. writing thefinal report;


     5. preparing the presentation of your work.