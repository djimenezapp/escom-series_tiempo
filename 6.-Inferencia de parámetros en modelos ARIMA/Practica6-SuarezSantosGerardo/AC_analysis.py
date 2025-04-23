import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(data_series):
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plot_acf(data_series, ax=plt.gca(), lags=40)
    plt.title('Autocorrelation Function (ACF)')
    
    plt.subplot(122)
    plot_pacf(data_series, ax=plt.gca(), lags=40)
    plt.title('Partial Autocorrelation Function (PACF)')
    
    plt.tight_layout()
    plt.show()