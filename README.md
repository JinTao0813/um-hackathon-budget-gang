# 📊 Backtesting — Python Library for Quantitative Trading and Flow Signal Strategy

**UM Hackathon 2025 – Domain 2: Quantitative Trading**

## 🌸 Final Round Canva Link

https://www.canva.com/design/DAGk-OIg1W4/bws2cLvcZ33hWWKck_ogwg/view?utm_content=DAGk-OIg1W4&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h90d57df12d

Backtesting is a powerful and modular Python library specifically built for cryptocurrency quantitative trading strategy development. Created during the UM Hackathon 2025, it empowers crypto traders and quantitative researchers to unlock insights from both numerical market data and textual sources. With Backtesting, users can analyze cryptocurrency market indicators, on-chain blockchain metrics, and sentiment from financial news to build predictive models. The library supports the full trading strategy pipeline, from data collection and model development to backtesting on historical data and a forward testing strategy evaluation. Whether you're working with Bitcoin, Ethereum, or other cryptocurrencies, Backtesting provides the tools needed to develop and test cutting-edge crypto trading strategies with precision.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-002A6D?style=for-the-badge&logo=python&logoColor=white)](https://www.statsmodels.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FF3C00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![TA-Lib](https://img.shields.io/badge/TA--Lib-00A4A6?style=for-the-badge&logo=python&logoColor=white)](https://www.ta-lib.org/)
[![hmmlearn](https://img.shields.io/badge/hmmlearn-FF8C00?style=for-the-badge&logo=python&logoColor=white)](https://github.com/hmmlearn/hmmlearn)
[![FinBERT](https://img.shields.io/badge/FinBERT-FF99CC?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/ProsusAI/finbert)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)

---

## 🚀 Key Features

| Category                   | Feature                           | Description                                                                                                                        |
| -------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 📥 Data Collection         | Crypto Market Indicators          | Collect historical price data for cryptocurrencies like Bitcoin using technical indicators such as RSI, MACD, and moving averages. |
|                            | On-Chain Metrics                  | Integrate blockchain-based metrics like MVRV, Netflow, SOPR, and more for deeper market insights.                                  |
|                            | Data Cleaning & Structuring       | Preprocess and format both price and on-chain data for seamless model ingestion.                                                   |
| 🤖 Machine Learning Models | Hidden Markov Models (HMM)        | Detect market regimes and latent states in financial time series. Useful for trend analysis and strategy switching.                |
|                            | Natural Language Processing (NLP) | Extract sentiment from financial news, tweets, or articles using transformer-based models (e.g., BERT).                            |
|                            | Long Short Term Model (LSTM)      | Predicts the price and identify the market trend. signals.                                                                         |
| 📈 Strategy Implementation | MarketRegimeStrategy              | Uses HMM to detect bullish/bearish/neutral market states and adjusts positions accordingly.                                        |
|                            | SentimentAnalysisStrategy         | Reacts to sentiment signals extracted from news and articles to guide trading decisions.                                           |
|                            | DeepPredictorStrategy             | Uses LSTM to generate the buy/sell/hold signals based on price predicted.                                                          |
|                            | FlowSignalStrategy                | Combines the signals from several strategies to produce the final decision for backtesting.                                        |
| 📊 Backtesting             | Historical Simulation             | Simulate trading strategies on past data with customizable rules.                                                                  |
|                            | Performance Metrics               | Automatically calculate Sharpe Ratio, Drawdown, Win/Loss ratio, and more.                                                          |
|                            | Strategy Evaluation               | Visualize cumulative returns and compare strategy outcomes.                                                                        |
| ⏱️ Forward Testing         | Paper Trading Mode                | Run strategies on recent or live data feeds to simulate real-time behavior.                                                        |

## 💻 Installation

```bash
pip install -r requirements.txt
```

Note: Recommended Python version: 3.11.9 due to compatibility with TensorFlow and other libraries.

## 📌 How to Run

The showcase of backtesting to evaluate a strategy is shown in finalSignalStrategy.ipynb
