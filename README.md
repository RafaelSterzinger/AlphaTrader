# Project Proposal
## Scientific Papers
For my project in Applied Deep Learning I chose to focus on Deep Reinforcement Learning (DRL) in the financial market or rather on the stock market.
The idea behind this proposal was to create a Deep Q Network (DQN) which can trade financial products from tech-companies, such as Google or Apple.
This topic seems to attract a great deal of attention, since there are dozens of scientific papers on sites like e.g. [arXiv.org](https://arxiv.org/) covering this problem.
Therefore, there are many directions in which this project might develop, but for the beginning I will use a simple DQN in combination with the following four papers:

* [Reinforcement Learning in Stock Trading](https://hal.archives-ouvertes.fr/hal-02306522/document)

* [Practical Deep Reinforcement Learning Approach forStock Trading](https://arxiv.org/pdf/1811.07522.pdf)

* [Deep Reinforcement Learning in Financial Markets](https://arxiv.org/abs/1907.04373)

* [Deep Reinforcement Learning for Foreign Exchange Trading](https://arxiv.org/abs/1908.08036)

This papers were mainly used to get an idea on how to preprocess financial data, design training and testing datasets and define a benchmark to evaluate the performance of the implemented agent. 

----
Other approaches, which were not used for now, but could be of future interest are the usage of Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN), with a focus on models with a Long Short-Term Memory (LSTM). 

**CNN's**

[Predict Forex Trend via Convolutional Neural Networks](https://arxiv.org/abs/1801.03018), [Conditional time series forecasting with convolutional neural networks](https://arxiv.org/abs/1703.04691), [Using Deep Learning Neural Networks and Candlestick Chart Representation to Predict Stock Market](https://arxiv.org/abs/1903.12258)

**RNN's**

[Stock Prices Prediction using Deep Learning Models](https://arxiv.org/abs/1909.12227), [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948), [Financial series prediction using Attention LSTM](https://arxiv.org/abs/1902.10877)

----
Another idea for the future is the inclusion of sentiment analysis in the model. Papers available on this topic are: 

* [Forex trading and Twitter: Spam, bots, and reputation manipulation](https://arxiv.org/abs/1804.02233) <br> => Research on the influence of Tweets on the market and whether to buy, hold or sell.

* [Listening to Chaotic Whispers: A Deep Learning Framework for News-oriented Stock Trend Prediction](https://arxiv.org/pdf/1712.02136) <br> => Mechanism to process recent news related to the stock market.
----
Another approach provides [this](https://arxiv.org/abs/1910.05137) paper, which tries to simulate the "whole stock market" in a multi agent system (MAS), where each agent learns individually and trades on its own. The collective behaviour of agents is then used to predict the market. This method might be out of the projects scope at the moment due to missing processing power and time, but might be of interest in future work.


## Topic
As already mentioned, this project will have a focus on __Reinforcement Learning (RL)__, especially in the context of stock trading and the prediction of this market using a DQN.

## Project Type
Concerning the project type, there are many options applicable. Types like **Bring your own data, Bring your own method** and **Beat the stars** can all be applied, since the project can evolve in many directions in the future. For example **Bring your own data** may be needed if future work focuses on the inclusion of sentiment analysis in the prediction. However if the project goes beyond the scope of this lecture, focus will be lied solely on DRL with a DQN agent, which will at least result in **Bring your own method**.

## Summary
* __Description and Approach__

    The goal of the project is to predict different stocks from different companies, such as Google or Apple.

    I will begin with standard DRL approaches listed on [SpinningUp](https://spinningup.openai.com/en/latest/user/algorithms.html) and their [Baseline Implementation](https://github.com/openai/baselines) to get an overview and a general practical understanding of this field as well as an insight in [Keras](https://keras.io/) or [PyTorch](https://pytorch.org/). Then I will try to use different approaches from the earlier mentioned used papers to predict the market with DRL.

    After a first working model has been implemented, it will be used as a baseline for further hyper parameter tuning and model variations. 
    
    For general comparison I will use a third party extension of the [OpenAI Gym Toolkit](https://github.com/openai/gym) called [AnyTrading](https://github.com/AminHP/gym-anytrading), which is a testing and training environment to compare trading approaches.

* __Dataset__

    To collect the necessary data, I will use the [GoogleFinance API](https://support.google.com/docs/answer/3093281) with a corresponding [Python module](https://pypi.org/project/googlefinance.get/). Depending on how the data are split into training and testing and if absolute prices or relative price changes will be used, there might be the necessity of preprocessing.

* __Work-Breakdown Structure__

__Work Description__

     (1) research topic and first draft
    
   First I focused my research on DRL with the idea of implementing an engine which masters the game Onitama, or any of the Atari-Games provided by the OpenAI Gym. Since Onitama seemed to go beyond the lectures scope and the Atari-Games seemed to be too simple, I chose the topic of forex trading, which would also use DRL. Luckily there is already a training and testing environment, called AnyTrading, provided by OpenAI, which means that I will have more time, focusing on the application of different methods. After chosing my project I focused on getting an overview of different solutions to this problem by reading parts of the scientific papers as well as watching some online lectures from MIT. Lastly, I wrote this description and started playing around with the technologies which will be used.
     
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
| setting up the environment and acquiring the datasets      | 3h                   | 6h        |
| designing and building an appropriate network              | 15h                  | 18.5h     |
| fine-tuning and varying that network                       | 15h                  | 14h       |
| updating readme                                            | 1h                   | 1h        |
| building an application to present the results             | 6h                   |           |
| writing the final report                                   | 5h                   |           |
| preparing the presentation of the project                  | 3h                   |           |

