# ai-fin

A repo for our COMP 598 project exploring applications of deep learning to finance.


## Data

#### Traditional

- [NYSE](https://www.kaggle.com/dgawlik/nyse)  
- [Istanbul SE](https://www.kaggle.com/uciml/istanbul-stock-exchange)
- [World News + Stock data](https://www.kaggle.com/aaron7sun/stocknews)

#### Crypto

- [Daily Prices all coins](https://www.kaggle.com/jessevent/all-crypto-currencies)
- [Another Daily Prices](https://www.kaggle.com/taniaj/cryptocurrency-market-history-coinmarketcap)


## Papers

- [Dual Attention Model]( https://arxiv.org/pdf/1704.02971.pdf)  
- [2020 review paper](https://arxiv.org/abs/2003.01859)  
- [Training a Neural Network with Financial Criterion](http://www.iro.umontreal.ca/~lisa/pointeurs/nncm.pdf)
- [1996 Classic Yoshua Paper](https://papers.nips.cc/paper/1996/file/1d72310edc006dadf2190caad5802983-Paper.pdf)  
  -  I think this paper presents results and general ideas related to paper above.
  -  They implement linear layers (5-3-1 and 5-3-2-1)
  -  Try to predict financial criteria instead of price. They claim it's better based on results in paper above.
  -  They explore the effects of sharing parameters of models (None, Some, All) for different stocks (hence the term Multi Task).
  -  Turns out sharing everything (i.e. having one model for all stocks) is much worse than the rest. You need some sort of specific training for different stocks. 
- [An adversarial approach](https://www.ijcai.org/Proceedings/2019/0810.pdf)  

## Textbooks
- [Advances in Financial Machine Learning (Marco Lopez De Prado)](https://www.amazon.ca/Advances-Financial-Machine-Learning-Marcos/dp/1119482089)
  -  Alot of good stuff on how to manipulate data, extract features, backtest correctly, etc
  -  Nothing on actual ML models or architectures though
## Code

- [Deep RL for Finance library](https://github.com/AI4Finance-LLC/FinRL-Library)
- [A VERY good repo with links to everything](https://github.com/firmai/financial-machine-learning)  
- [Dual Attention Model Git](https://github.com/Seanny123/da-rnn)  
- [Understandable RL algo implementations](https://github.com/higgsfield/RL-Adventure-2)



## Propostions

- Explore transfer learning of RL in the context of finance. ie, train a model on traditional markets and then apply the model to crypto trading 
    - Do we need to fine tune? 
    - If so to what extent?
    - Is it more efficient to adapt or retrain?



