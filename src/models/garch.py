from arch import arch_model
import pandas as pd
import numpy as np

def garch_model(returns: pd.Series, p: int = 1, q: int = 1):
    scale_factor = 100
    scaled_returns = returns * scale_factor

    model = arch_model(scaled_returns, vol='GARCH', p=p, q=q, rescale=False)
    result = model.fit(disp='off')
    
    garch_vol = result.conditional_volatility / scale_factor
    
    return result, garch_vol

def garch_model_forecast(returns: pd.Series, p: int = 1, q: int = 1) -> float:
    scale_factor = 100
    scaled_returns = returns * scale_factor

    model = arch_model(scaled_returns, vol='GARCH', p=p, q=q, rescale=False)
    result = model.fit(disp='off')

    forecast_variance = result.forecast(horizon=1).variance

    if isinstance(forecast_variance, pd.DataFrame):
        forecast_var_value = forecast_variance.iloc[-1, 0]
    else:
        forecast_var_value = forecast_variance.iloc[-1]
    
    forecast_var_value = forecast_var_value / (scale_factor ** 2)
    forecast_vol = np.sqrt(forecast_var_value)
    
    return float(forecast_vol)

