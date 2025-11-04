from arch import arch_model
import pandas as pd
import numpy as np

def garch_model(returns: pd.Series, p: int = 1, q: int = 1):
    """
    Fit a GARCH(p, q) model to the returns.
    Args:
        returns: pd.Series, the returns of the asset
        p: int, the order of the GARCH model (default: 1)
        q: int, the order of the ARCH model (default: 1)
    Returns:
        tuple: (result, garch_vol) where result is the fitted model and garch_vol is the conditional volatility
    """
    # Rescale returns to avoid scaling issues (recommended by arch library)
    # Returns are typically small (e.g., 0.001), so multiply by 100
    scale_factor = 100
    scaled_returns = returns * scale_factor
    
    # Fit the model with rescaled data
    model = arch_model(scaled_returns, vol='GARCH', p=p, q=q, rescale=False)
    result = model.fit(disp='off')
    
    # Get conditional volatility and rescale back to original units
    # Volatility is in same units as returns, so divide by scale_factor
    garch_vol = result.conditional_volatility / scale_factor
    
    return result, garch_vol

def garch_model_forecast(returns: pd.Series, p: int = 1, q: int = 1) -> float:
    """
    Forecast the volatility of the asset using a GARCH(p, q) model.
    Args:
        returns: pd.Series, the returns of the asset
        p: int, the order of the GARCH model (default: 1)
        q: int, the order of the ARCH model (default: 1)
    Returns:
        float, the forecast of the volatility (in original units)
    """
    # Rescale returns to avoid scaling issues
    scale_factor = 100
    scaled_returns = returns * scale_factor
    
    # Fit the model with rescaled data
    model = arch_model(scaled_returns, vol='GARCH', p=p, q=q, rescale=False)
    result = model.fit(disp='off')
    
    # Get forecast and rescale back to original units
    # Variance scales with square of returns, so divide by scale_factor^2, then take sqrt
    forecast_variance = result.forecast(horizon=1).variance
    # Extract the last value (forecast variance)
    if isinstance(forecast_variance, pd.DataFrame):
        forecast_var_value = forecast_variance.iloc[-1, 0]
    else:
        forecast_var_value = forecast_variance.iloc[-1]
    
    forecast_var_value = forecast_var_value / (scale_factor ** 2)
    forecast_vol = np.sqrt(forecast_var_value)
    
    return float(forecast_vol)