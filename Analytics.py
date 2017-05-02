import pandas as pd
import pandas_datareader.data as web
import numpy as np
import math
from urllib2 import Request, urlopen
import matplotlib.pyplot as plt
import json
import datetime
'''
 A Class for Quant portfolio metrics
'''

class Portfolio():


    def __init__(self,equities,weights,start,end,money): #Portfolio Construction
        self.equities = equities
        self.weights = weights
        self.start = start #'2014-12-01' format
        self.end = end
        self.money = money #Initial money


    def Portfolio_Adj_Close(self): #Adj Close
        d = {}
        for ticker in self.equities:
            d[ticker] = web.DataReader(ticker, "yahoo", self.start, self.end)
        pan = pd.Panel(d)
        df_adj_close = pan.minor_xs('Adj Close')
        self.df_adj_close = df_adj_close
        return df_adj_close
    def Portfolio_Volume(self): #Volume
        d = {}
        for ticker in self.equities:
            d[ticker] = web.DataReader(ticker, "yahoo", self.start, self.end)
        pan = pd.Panel(d)
        df_volume = pan.minor_xs('Volume')
        self.df_volume = df_volume
        return df_volume


    def Portfolio_Value(self):
        self.df_adj_close = self.Portfolio_Adj_Close()
        normed = self.df_adj_close/self.df_adj_close.ix[0,:] #Norming the prices
        alloced = normed*self.weights
        pos_vals = alloced*self.money
        series_portfolio_value = pos_vals.sum(axis=1)
        self.series_portfolio_value = series_portfolio_value
        return series_portfolio_value


    def Plot_Portfolio_Value(self):
        plt.style.use('ggplot')
        port_val = self.Portfolio_Value()
        port_val = port_val.to_frame()
        dates =  port_val.index
        port_val = port_val.values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_title('Portfolio Value')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        plt.plot(dates,port_val)
        #TROUBLE HERE
        fig.savefig('static/plot_port_val.png')
        plt.close()
        return


    def daily_returns(self):
        self.series_portfolio_value = self.Portfolio_Value()
        daily_returns = (self.series_portfolio_value/self.series_portfolio_value.shift(1))-1
        daily_returns = daily_returns[1:]
        self.df_daily_returns = daily_returns
        return daily_returns


    def cumulative_returns(self):
        self.series_portfolio_value = self.Portfolio_Value()
        #print self.series_portfolio_value
        cumulative_returns = (self.series_portfolio_value[-1]/self.series_portfolio_value[0])-1
        self.df_cumulative_returns = cumulative_returns
        return cumulative_returns


    def average_daily_returns(self):
        average_daily_returns = self.daily_returns().mean()
        self.df_average_daily_returns = average_daily_returns
        return average_daily_returns


    def volatility(self):
        volatility = self.daily_returns()
        volatility = volatility.std()
        self.volatility = volatility
        return volatility


    def Sharpe_Ratio(self,rf=0):
        #rf = 0 #The Risk Free Rate
        samples_per_year = 252
        mean_avg_daily_rets = (self.average_daily_returns() - rf).mean()
        vol = self.volatility()
        sharpe = np.sqrt(samples_per_year)*(mean_avg_daily_rets/vol)
        self.sharpe_ratio = sharpe
        return sharpe


    #TODO: Check to make sure that this is the correct value
    def Conditional_Sharpe_Ratio(self,expected_return, risk_free_rate=0):
        cSR = (expected_return - risk_free_rate) / self.Condtition_Value_At_Risk()
        self.conditional_sharpe_ratio = cSR
        return cSR


    def Treynor_Ratio(self,excess_returns,risk_free_rate=0):
        treynor_ratio = (excess_returns - risk_free_rate) / self.Beta()
        self.treynor_ratio = treynor_ratio
        return treynor_ratio


    def Information_Ratio(self,benchmark='SPY'):
        Market_Portfolio = Portfolio([benchmark],1, self.start, self.end,self.money)
        market_returns = np.array(Market_Portfolio.daily_returns())
        returns = np.array(self.daily_returns())
        difference = returns - market_returns
        vol_differnece = difference.std()
        information_ratio = np.mean(difference) / vol_differnece
        self.information_ratio = information_ratio
        return information_ratio


    def Modigliani_Ratio(self,expected_returns, benchmark='SPY', risk_free_rate=0):
        returns = np.array(self.daily_returns())
        Market_Portfolio = Portfolio([benchmark],1, self.start, self.end,self.money)
        market_returns = np.array(Market_Portfolio.daily_returns())
        arry = np.empty(len(returns))
        arry.fill(risk_free_rate)
        difference = returns - arry
        benchmark_difference = market_returns - arry
        modigliani_ratio = (expected_returns - risk_free_rate) * (difference.std() / benchmark_difference.std()) + risk_free_rate
        self.modigliani_ratio = modigliani_ratio
        return modigliani_ratio


    def Beta(self):
        Market_Portfolio = Portfolio(['SPY'],1, self.start, self.end,self.money)
        market_returns = Market_Portfolio.daily_returns() #Market Returns
        self.market_returns = market_returns
        portfolio_returns = self.daily_returns()  #Our Portfolio Returns
        covariance = np.cov(portfolio_returns,market_returns)[0][1]
        variance = np.var(market_returns)
        beta = covariance / variance
        self.beta = beta
        return beta


    def Alpha(self,rf=0): #TODO: Optionally make risk free rate a parameter
        #Also called Jensen's Alpha
        rf = 0
        beta = self.Beta()
        Market_Portfolio = Portfolio(['SPY'],1, self.start, self.end,self.money)
        market_returns = Market_Portfolio.cumulative_returns()
        portfolio_returns = self.cumulative_returns()
        alpha = portfolio_returns - (rf + (market_returns-rf)*beta)
        self.alpha = alpha
        return alpha



    def Omega_Ratio(self,expected_returns,risk_free_rate=0,target=0):
        lpm = self.Lower_Partial_Moment(target,order=1)
        omega = (expected_returns-risk_free_rate)/lpm
        self.omega = omega
        return omega


    def Sortino_Ratio(self,expected_returns,risk_free_rate=0,target=0):
        lpm = math.sqrt(self.Lower_Partial_Moment(target,order=2))
        sortino_ratio = (expected_returns - risk_free_rate) / lpm
        self.sortino_ratio = sortino_ratio
        return sortino_ratio


    def Calmar_Ratio(self,expected_returns,risk_free_rate=0):
        max_draw_down = self.Max_Draw_Down()
        calmar_ratio = (expected_returns - risk_free_rate) / max_draw_down
        self.calmar_ratio = calmar_ratio
        return calmar_ratio

    #TODO: Slow due to drawdown calculation
    def Sterling_Ratio(self,expected_returns,periods,risk_free_rate=0):
        average_draw_down = self.Average_Drawdown(periods)
        sterling_ratio = (expected_returns - risk_free_rate) / average_draw_down
        return sterling_ratio


    #TODO: Slow due to drawdown caluclation
    def Burke_Ratio(self,expected_returns,periods, risk_free_rate=0):
        average_drawdown_squared = math.sqrt(self.Average_Drawdown_Squared(periods))
        burke_ratio = (expected_returns - risk_free_rate) / average_drawdown_squared
        self.burke_ratio = burke_ratio
        return burke_ratio


    #TODO: Test this to make sure it is working, giving difference between expected_returns and risk_free_rate
    #TODO: lpm keeps equaling 1
    def Kappa_Three_ratio(self,expected_returns,risk_free_rate=0,target=0):
        lpm = math.pow(self.Lower_Partial_Moment(target,order=3), float(1/3))
        kappa_three_ratio = (expected_returns - risk_free_rate) / lpm
        self.kappa_three_ratio = kappa_three_ratio
        return kappa_three_ratio


    def Lower_Partial_Moment(self,target=0,order=1):
        returns = np.array(self.daily_returns()) #an np array so we can use .clip()
        threshold_array = np.empty(len(returns))
        threshold_array.fill(target)
        diff = threshold_array - returns
        diff = diff.clip(min=0)
        lpm = np.sum(diff ** order) / len(returns)
        self.lower_partial_moment = lpm
        return lpm


    def Higher_Partial_Moment(self,target=0,order=1):
        returns = np.array(self.daily_returns())
        threshold_array = np.empty(len(returns))
        threshold_array.fill(target)
        diff = returns - threshold_array
        diff = diff.clip(min=0) # For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
        hpm = np.sum(diff ** order) / len(returns)
        self.higher_partial_moment = hpm
        return hpm


    def Gain_Loss_Ratio(self,target=0):
        hpm = self.Higher_Partial_Moment(target,1)
        lpm = self.Lower_Partial_Moment(target,1)
        gain_loss_ratio = hpm / lpm
        self.gain_loss_ratio = gain_loss_ratio
        return gain_loss_ratio


    def Upside_Potential_Ratio(self,target=0):
        hpm = self.Higher_Partial_Moment(target,order=1)
        lpm = math.sqrt(self.Lower_Partial_Moment(target,order=2))
        upside_potential_ratio = hpm / lpm
        self.upside_potential_ratio = upside_potential_ratio
        return upside_potential_ratio


    def Value_At_Risk(self):
        returns = np.array(self.daily_returns())
        alpha = self.Alpha()
        sort_returns = np.sort(returns)
        indx = alpha*(len(sort_returns))
        indx = int(indx)
        var = abs(sort_returns[indx])
        self.value_at_risk = var
        return var


    def Excess_Return_On_Value_At_Risk(self,excess_return,risk_free_rate=0):
        alpha = self.Alpha()
        returns = np.array(self.daily_returns())

        sorted_returns = np.sort(returns)
        indx = int(self.Alpha()*len(sorted_returns))
        vari = abs(sorted_returns[indx])
        ervar = (excess_return - risk_free_rate) / vari
        self.excess_return_on_value_at_risk = ervar
        return ervar


    def Condtition_Value_At_Risk(self):
        returns = np.array(self.daily_returns())
        alpha = self.Alpha()
        sort_returns = np.sort(returns)
        indx = alpha*(len(sort_returns))
        indx = int(indx)
        sigma_var = sort_returns[0]
        for i in range(1,indx):
            sigma_var += sort_returns[i]
        mcvar = abs(sigma_var/indx)
        self.condition_value_at_risk = mcvar
        return mcvar


    def Drawdown(self,tau):
        returns = np.array(self.daily_returns())
        s = [100]
        for i in range(len(returns)):
            s.append(100 * (1 + returns[i]))
        values = np.array(s)
        pos  = len(values) - 1
        pre = pos - tau
        drawdown = float('+inf')
        while pre >= 0:
            dd_i = (values[pos]/values[pre])-1
            if dd_i < drawdown:
                drawdown = dd_i
            pos, pre = pos - 1, pre - 1
        drawdown = abs(drawdown)
        self.drawdown = drawdown
        return drawdown


    def Max_Draw_Down(self): #TODO: Very Slow If there are a lot of returns
        returns = np.array(self.daily_returns())
        max_drawdown = float('-inf')
        for i in range(0,len(returns)):
            drawdown_i = self.Drawdown(i)
            if drawdown_i > max_drawdown:
                max_drawdown = drawdown_i
        max_drawdown = abs(max_drawdown)
        self.max_drawdown = max_drawdown
        return max_drawdown


    def Average_Drawdown(self,periods):
        drawdowns = []
        returns  = self.daily_returns()
        for i in range(0,len(returns)):
            drawdown_i = self.Drawdown(i)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_drawdown = abs(drawdowns[0])
        for i in range(1,periods):
            total_drawdown += abs(drawdowns[i])
        average_drawdown = total_drawdown / periods
        self.average_drawdown = average_drawdown
        return average_drawdown


    def Average_Drawdown_Squared(self,periods):
        drawdowns = []
        returns  = self.daily_returns()
        for i in range(0,len(returns)):
            drawdown_i = math.pow(self.Drawdown(i),2.0)
            drawdowns.append(drawdown_i)
        drawdowns = sorted(drawdowns)
        total_drawdown = abs(drawdowns[0])
        for i in range(1,periods):
            total_drawdown += abs(drawdowns[i])
        average_drawdown_squared = total_drawdown / periods
        self.average_drawdown_squared = average_drawdown_squared
        return average_drawdown_squared


    def Portfolio_Price_To_Book(self):
        ptb_list = []

        for stock in self.equities:
            yhf_link = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (stock, 'p6')
            req = Request(yhf_link)
            resp = urlopen(req)
            ptb = float(resp.read().decode().strip())
            ptb_list.append(ptb)
        average_price_to_book = np.mean(ptb_list)
        self.average_price_to_book = average_price_to_book
        return average_price_to_book


    def Portfolio_Price_to_Earnings(self):
        pe_list = []

        for stock in self.equities:
            yhf_link = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (stock, 'r')
            req = Request(yhf_link)
            resp = urlopen(req)
            ptb = float(resp.read().decode().strip())
            pe_list.append(ptb)
        average_price_to_earnings = np.mean(pe_list)
        self.average_price_to_book = average_price_to_earnings
        return average_price_to_earnings

    def Portfolio_PEG(self):
        peg_list = []

        for stock in self.equities:
            yhf_link = 'http://finance.yahoo.com/d/quotes.csv?s=%s&f=%s' % (stock, 'r5')
            req = Request(yhf_link)
            resp = urlopen(req)
            ptb = float(resp.read().decode().strip())
            peg_list.append(ptb)
        average_PEG = np.mean(peg_list)
        self.average_PEG = average_PEG
        return average_PEG

if __name__ == '__main__':
    tech = Portfolio(['AAPL','GILD','MSFT','KO'],[.25,.25,.25,.25],'2010-01-01','2015-12-01',100000)
    x = tech.cumulative_returns()
    print x
    
    sharpe = tech.Sharpe_Ratio()
    print sharpe