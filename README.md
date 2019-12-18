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

    The datasets for training and testing will be acquired from [Yahoo! Finance](https://finance.yahoo.com/), focusing on tech companies like Google or Apple. However, any other stock data would work as well. For the pre-processing of this data, I will start evaluating the agent on non pre-processed data, followed by different scaling methods, such as Sigmoid, MinMax or Standard.

* __Work-Breakdown Structure__
     
| Individual Task                                            | Time estimate        | Time used |
|------------------------------------------------------------|----------------------|-----------|
| research topic and first draft                             | 5h                   | 13h       |
| setting up the environment and acquiring the datasets      | 3h                   | 7h        |
| designing and building an appropriate network              | 15h                  | 19h       |
| fine-tuning and varying that network                       | 15h                  | 15h       |
| building an application to present the results             | 6h                   |           |
| writing the final report                                   | 5h                   |           |
| preparing the presentation of the project                  | 3h                   |           |

## Implementation
### Error Metric
* __Error Metric__ <br> 
Every agent and their variation of pre-processing and structure, will be trained for 650 epochs on the trainings dataset (AAPL_train.csv). 
Therefore, different approaches can be evaluated and compared using the average profit as well as the average reward of the last 50 epochs (600-650). <br><br>
__Reward__ is defined by the capability to correctly predict the direction of stock price of the following day. 
For example, if the price falls and the agent bid on falling prices (SHORT), it will receive a positive reward or if the price falls and the agent bid on rising prices (LONG), it will receive a negative reward, consisting of the price difference.<br><br>
__Profit__ is defined by the price difference between two time steps, where the agent chose to change its opinion on the trend, switching from LONG to SHORT or the other way around.
This definition implies a trade, where the agent e.g. sells all its LONG-positions and buys as much SHORT-positions as possible, to not lose money.<br><br>
This metric is used to verify that the agent is actually making progress. Since this verification is only used on the trainings dataset, it does not give an estimation on the real-life performance.
Thus, a test suite was implemented to compare models on unseen data and compare them by earned profit and reward on a given test set (AAPL_test.csv)

* __Error Metric Target__ <br>
First benchmarks of the implemented agent were quite misleading, resulting in an average profit of __0.477__ and an average reward of __3.568__. 
Thus, I set my target to reach at least an average profit of __1__, which would mean
that the agent is at least profitable on the trainings set. After many iterations of
 adjusting hyper parameters and changing the model and still resulting in really bad and random performance, 
 I took a closer look on the implementation of the used environment, called AnyTrading. After a short observation, I felt completely unsatisfied
 with the implementation and therefore defined my own calculations of reward and profit. This change finally gave me the impression that my agent is making progress. Thus, earlier saved models and plots are not comparable to newer ones.
 After the change the target goal of 1 was quite simple to archive and is therefore not really representative.

* __Error Metric Achievement__ <br>
The following table displays the performance results of the last 7 agent variations, which all performed better than the target of __1__.

|Average Profit| Average Reward|
|--------------|---------------|
|19.794        |984.336        |
|2.763         |507.834        | 
|6.313         |207.225        |
|22.684        |992.019        |
|8.445         |730.180        |
|15.148        |474.520        |
|5.843         |349.651        |

The following plot shows the average profit by episode<br>
![Plot of the average profit by episode](https://github.com/RafaelSterzinger/Applied-Deep-Learning/blob/master/plots/profit18_16_34.png)

and the average reward of the best model.<br>
![Plot of the average profit by episode](https://github.com/RafaelSterzinger/Applied-Deep-Learning/blob/master/plots/reward18_16_34.png)

Since the evaluation of the agent on the trainings set is not that interesting and is only used to verify that the agent is actually learning something, I will provide some plots, which show the performance of the model on unseen data.<br><br>

Green dots are time steps, where the agent decided to go LONG<br>
Red dots are time steps, where the agent decided to go SHORT

Plot of a model trained on AAPL, tested on GOOG<br>
![Plot of a model trained on AAPL, tested on GOOG](https://github.com/RafaelSterzinger/Applied-Deep-Learning/blob/master/plots/trades_model_18_16_34_AAPL_on_GOOG.png)

Plot of a model trained on GOOG, tested on GOOG<br>
![Plot of a model trained on GOOG, tested on GOOG](https://github.com/RafaelSterzinger/Applied-Deep-Learning/blob/master/plots/trades_model_18_17_06_GOOG_on_GOOG.png)

Plot of a model trained on GOOG, tested on AAPL<br>
![Plot of a model trained on GOOG, tested on AAPL](https://github.com/RafaelSterzinger/Applied-Deep-Learning/blob/master/plots/trades_model_18_17_06_GOOG_on_AAPL.png)

### Changelog
