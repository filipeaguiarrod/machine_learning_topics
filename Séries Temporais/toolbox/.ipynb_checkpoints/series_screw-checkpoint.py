from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import matplotlib.pyplot as plt


def adfuller_eval_rand(x):
    
    result = adfuller(x)[1]
    
    if result >= (5/100):
        
        return f'p_value: {result} não podemos rejeitar Ho, de que é uma Random Walk'
    
    else:
        
        return f'p_value: {result} rejeita-se Ho, não é uma Random Walk'
    
    
def adfuller_eval_sta(x):
    
    result = adfuller(x)[1]
    
    if result >= (5/100):
        
        return f'p_value: {result} não podemos rejeitar Ho. Série Não Estacionária'
    
    else:
        
        return f'p_value: {result} rejeita-se Ho. Esta séries é estacionária'
    

def acf_pacf(df,lags):
    
    '''
    Retorna acf e pacf para comparação e entendimento se é AR, MA ou ARMA
    '''

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

    # Plot the ACF of df
    plot_acf(df, lags=lags, zero=False, ax=ax1)

    # Plot the PACF of df
    plot_pacf(df, lags=lags, zero=False, ax=ax2)

    plt.show()