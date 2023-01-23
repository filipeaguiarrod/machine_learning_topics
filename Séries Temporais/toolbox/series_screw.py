from statsmodels.tsa.stattools import adfuller


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