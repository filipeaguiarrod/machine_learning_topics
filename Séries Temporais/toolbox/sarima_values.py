from typing import Union
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product

def optimize_SARIMAX(endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int, suppress_warnings: bool = True) -> pd.DataFrame:
    results = []
    for order in tqdm_notebook(order_list):
        try:
            model = SARIMAX(endog,order=(order[0], d, order[1]),
            seasonal_order=(order[2], D, order[3], s),
            simple_differencing=False, suppress_warnings=suppress_warnings).fit(disp=False)
        except:
            continue
        aic = model.aic
        results.append([order, aic,model.summary().tables[2].data[1]])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q,P,Q)', 'AIC','Residuals']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC',ascending=True).reset_index(drop=True)
    return result_df
