# PyQuant
A class for quantitative portfolio metrics

Usage: 
- First instantiate a portfolio with a list of tickers, weights for each ticker, a start date, an end date, and the value of the portfolio.


    tech = Portfolio(['AAPL','GILD','MSFT','KO'],[.25,.25,.25,.25],'2010-01-01','2015-12-01',100000)

- Then you can use the instantiated portfolio and call the above methods, for example you can access the cumulative returns of the portfolio and the Sharpe Ratio as follows:


    x = tech.cumulative_returns()
    print x

    sharpe = tech.Sharpe_Ratio()
    print sharpe
