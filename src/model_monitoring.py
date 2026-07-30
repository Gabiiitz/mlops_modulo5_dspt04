import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from ft_engineering import load_and_clean_data

def monitor_data_drift():
    """
    esta funcion compara los datos de referencia con datos nuevos
    para ver si las distribuciones cambiaron (data drift) y crea un grafico
    """
    X, _ = load_and_clean_data()
    
    # simulamos datos de referencia (80%) y datos nuevos (20%)
    ref_data = X.sample(frac=0.8, random_state=42)
    prod_data = X.drop(ref_data.index)
    
    features_to_monitor = ['capital_prestado', 'salario_cliente', 'puntaje_datacredito', 'saldo_total']
    
    drift_results = {}
    
    for col in features_to_monitor:
        ref_series = ref_data[col].dropna()
        prod_series = prod_data[col].dropna()
        
        # prueba de kolmogorov-smirnov
        stat, p_value = ks_2samp(ref_series, prod_series)
        
        has_drift = p_value < 0.05
        drift_results[col] = {
            'statistic': stat,
            'p_value': p_value,
            'drift_detected': has_drift
        }
        
        print(f"variable '{col}': p-value = {p_value:.4f} | drift detectado = {has_drift}")

    # graficamos
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(features_to_monitor, 1):
        plt.subplot(2, 2, i)
        plt.hist(ref_data[col].dropna(), bins=20, alpha=0.5, label='referencia', density=True, color='blue')
        plt.hist(prod_data[col].dropna(), bins=20, alpha=0.5, label='produccion', density=True, color='orange')
        plt.title(f'drift en {col}\n(drift: {drift_results[col]["drift_detected"]})')
        plt.legend()
        plt.grid(True)
        
    plt.tight_layout()
    plt.savefig('data_drift_monitoring.png')
    plt.show()

if __name__ == '__main__':
    monitor_data_drift()