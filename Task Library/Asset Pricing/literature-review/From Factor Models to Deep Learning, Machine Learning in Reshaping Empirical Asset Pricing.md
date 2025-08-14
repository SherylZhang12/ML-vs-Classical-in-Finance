
### 1. Paper Positioning and Contribution Overview

This paper systematically reviews the application of ML/AI in **empirical asset pricing**: it first reviews the foundation and limitations of traditional factor models, then discusses the role of ML in **risk premium estimation** and **portfolio optimization**, as well as innovations such as **dimensionality reduction, missing value imputation, alternative data, multimodality, denoising, and handling non-IID data**. Finally, it points out challenges such as data availability, structural changes, overfitting, interpretability, and compliance.

### 2. Problem Setup  (Risk Premium and Conditional Expectation)

- The core objective is to estimate the **excess return** of an asset (relative to the risk-free rate). Definition:  
    $( y_{i,t} = \mathbb{E}[y_{i,t}] + \varepsilon_{i,t} )$,   (1)
    where $( \mathbb{E}[\varepsilon_{i,t}] = 0 )$. The conditional expectation is often written as an unknown function $( g(\cdot) )$ of features $( x_{i,t} )$:  
    $( y_{i,t} = g(x_{i,t-1};\theta) + \varepsilon_{i,t} )$.  (2)
    Here $( X )$ usually contains **firm-specific characteristics** and **macroeconomic factors**.
    

### 3. Traditional Factor Models (Statistical Paradigm Under the Same Objective)

- Classic **linear factor regression**:  
    $( y_{i,t} = \alpha_{i,t-1} + \beta'_{i,t-1} f_t + \varepsilon_{i,t} )$.  (3)
    Intuition: replace high-dimensional features $( X )$ with low-dimensional (often **latent**) factors $( f_t )$ to explain cross-sectional differences in returns.
    
- Two ways to obtain latent factors:  
    (i) **Sort-based factor construction** (e.g., Fama–French five factors: size, value, profitability, investment). However, the broad “factor zoo” still fails to fully explain $( \alpha = 0 )$.  
    (ii) **Extract factors from return panels** (e.g., PCA). To capture **time-varying** factor loadings, IPCA sets:  
    $( y_{i,t} = x_{i,t-1} \beta_i f_t + \varepsilon_{i,t} )$,  (4)
    where $( \beta \in \mathbb{R}^{N\times K} )$, $( f_t \in\mathbb{R}^K )$ are to be estimated, $\epsilon_{i,t}$ is a composite error term that includes $\alpha_{i,t-1}$
    
- **Key limitations**:
    
    1. As the number of factors increases, equation (4) has more free parameters, and traditional regression estimation **loses efficiency**;
        
    2. It relies on **linear** approximations, while theoretical asset pricing suggests nonlinear dynamics in returns. Thus, more flexible tools are needed to handle high dimensionality and nonlinearity.
        

### 4. ML/AI-Enhanced Risk Premium and Price Prediction (ML Paradigm Under the Same Objective)

- **Direct learning of $( g(\cdot) )$** through empirical risk minimization:  
    $( \hat g = \arg\min_{g\in \mathcal{G}} \sum_{i=1}^N \sum_{t=1}^T (y_{i,t} - g(x_{i,t-1}))^2 )$.  (5)
    Applied to **cross-sectional return prediction**; can also extend to **time series** and **direction/ranking** predictions.
    
- **Model families**: from linear regression to nonlinear tree models (RF/GBRT) and deep neural networks (DNN), covering equities, cryptocurrencies, futures, and options.
    
- **Evaluation metrics** (by task):
    
    - Regression: RMSE/MAE/MAPE;
        
    - Classification: Accuracy/Precision/Recall/F1/MCC (MCC is suited for **class imbalance**);
        
    - Ranking: MAP/MRR/NDCG (definitions given in the paper’s Table 1).
        
- **Why ML outperforms under the same objective** (per paper): it can model **nonlinearities**, handle **high-dimensional data** with **factor/feature selection** ability, and integrate **nontraditional data sources** (text, images, audio, etc.).
    

### 5. Time Series and Spatio-Temporal Models (Method-Level Comparison Under the Prediction Objective)

- **Time series (Temporal)**: predict future from lagged inputs:  
    $( y_{i,t}, y_{i,t+1},\ldots = g_t(x_{i,t-1}, x_{i,t-2},\ldots, x_{i,1}) )$.  (6)
    In practice, the field has shifted from ARIMA/VAR to LSTM/RNN, then to **N-BEATS, TS-Mixer** (pure MLP models), and introduced **multi-scale** (Fourier, wavelet, downsampling) to capture short-, medium-, and long-term structures.
    
- **Spatio-temporal**: simultaneously capture **network linkages** (industry, supply chain, holdings, correlations) and temporal dynamics. General form:  
    $( y_t, y_{t+1},\ldots = g_t[,g_s(X_{t-1}), g_s(X_{t-2}),\ldots, g_s(X_1),] )$.  (7)
    Here $( g_s )$ is often implemented via **GNN/GAT/hypergraph attention**, then coupled with LSTM/GRU temporal modules, improving prediction and denoising.
    

### 6. Portfolio Optimization: Traditional MPT vs. Supervised Learning & Reinforcement Learning

- **Traditional MPT (same utility maximization objective)**:  
    $( w^{\ast} = \frac{1}{\gamma} \Sigma^{-1} \mu )$    (8)
    where $( \mu )$ and $( \Sigma )$ are the return mean and covariance, and $( \gamma )$ is risk aversion. Two approaches: estimate $( \mu, \Sigma )$ first, then solve $( w )$; or **directly parameterize weights** for optimization.
    
- **Supervised learning (ML path under same objective)**: first perform **regression/ranking/direction** prediction, then construct **long-short portfolios** based on predictions; common weights are **equal-weight** $( w_{i,t} = 1/N )$ or **score-weighted** $( w_{i,t} = v_{i,t} / \sum v_{i,t} )$.
    
- **Reinforcement learning RL (directly learn optimal weights)**: state includes historical prices $( x_t )$ and previous weights $( w_{t-1} )$, action is current weights $( w_t )$, and **value function**:  
    $( Q^{\pi}(s,a) = \mathbb{E}[\sum_{i=t}^{\infty} \gamma_i r_{t+i} ,|, s_t = x_t, a_t = w_{t-1}] )$.  (9)
    Methods such as EIIE use **online mini-batch training** and zero-impact assumptions in multi-market validation; later work introduces **attention**, **hierarchical execution and transaction costs**, **imitation learning**, and **contrastive learning**.
    

### 7. Methodological Innovations in Asset Pricing

- **Dimensionality reduction** (factor zoo → interpretability and robustness): traditional **PLS/PCA/LASSO**; more recently **autoencoders/variational autoencoders (VAE)** to learn latent pricing factors, integrating dynamic factors or diffusion structures to enhance robustness under noise.
    
- **Missing value imputation**: traditional “deletion/mean/zero fill” biases results; recent methods borrow from **recommender systems** (e.g., **coupled matrix factorization**) to fill analyst forecasts, use **Transformers** for feature imputation, and **tensor completion** for spatio-temporal financial data.
    
- **Alternative/multimodal data**: integrating **images** (treating time series as images/videos), **text & social media**, **audio** (earnings calls) with traditional data; advances in Transformers/LLMs push deeper multimodal modeling (e.g., VolTAGE integrating text+audio+graph networks to predict volatility).
    
- **Denoising and non-IID adaptation**: **contrastive learning** improves representation robustness; **Mixture-of-Experts (MoE)** uses a router to activate experts by market state, addressing **non-independence** and structural change (e.g., TRA, PASN).
    

### 8. Challenges and Directions

Overall challenges include **overfitting**, **interpretability**, **noise**, and **compliance** (GDPR/MiFID II/Basel III). The paper emphasizes:

- **Data availability and lack of unified benchmarks**, and
    
- **Market dynamics, structural breaks, and no-arbitrage causing model signals to decay quickly** (suggesting online learning and meta-learning as possible solutions).
    


### 9. Key Comparison: Traditional Statistics vs. ML/AI Under the Same Financial Objective

**A. Estimating Conditional Expectation/Risk Premium $( g(\cdot) )$ (Equations (2)/(5))**

- Traditional: approximates via **linear factors** (Equations (3)/(4)), requiring prior factor construction or factor extraction from returns; when factors are many or structure is nonlinear, risk of **underfitting/inefficiency** is high.
    
- ML/AI: **directly fits $( g )$**, using tree ensembles/DNNs to handle **nonlinearity+high dimensionality**, with built-in **feature selection**, and capable of integrating **text/image/audio** and other sources.
    
- Assessment: with the same data and goal, ML models are stronger in **flexibility** and **data integration**, but require attention to **overfitting and robust evaluation**.
    

**B. Temporal Dependence and Cross-Asset Linkages (Equations (6)/(7))**

- Traditional: mostly single-asset time series or cross-sectional regression, **hard to capture** both temporal and network structures simultaneously.
    
- ML/AI: **temporal deep models (LSTM/N-BEATS/TS-Mixer)** + **GNN/GAT/hypergraph** for **spatio-temporal** modeling, improving representation of short/long-term patterns and cross-asset linkages.
    

**C. Portfolio Optimization Under the Same Objective (Maximizing Expected Utility/Return-Risk Tradeoff)**

- Traditional (Equation (8)): estimate $( \mu, \Sigma )$ first, then solve $( w^* )$.
    
- Supervised learning: predict **returns/direction/ranking** first, then allocate based on rules (equal-weight/score-weight).
    
- Reinforcement learning (Equation (9)): **directly output weight policies**, incorporating **transaction costs/execution** and **attention/hierarchical/imitation** enhancements.
    
- Assessment: under the same return-risk objective, RL avoids error propagation from estimating $( \mu, \Sigma )$, suitable for **sequential decision-making**, but faces **stability and generalization** challenges.


