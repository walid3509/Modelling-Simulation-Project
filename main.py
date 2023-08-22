import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter


def read_data(file_name):
    data = pd.read_csv(file_name)
    return data

def fit_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def monte_carlo_projection(model, target_year, data, n_simulations=1000):
    last_year = pd.to_datetime(data['date'].iloc[-1]).year
    n_years = target_year - last_year
    preds = []

    X = np.array(data.index).reshape(-1, 1)
    y = data['GMSL']

    for _ in range(n_simulations):
        noise = np.random.normal(0, y.std(), len(data))
        sim_data = y + noise
        sim_model = fit_linear_regression(X, sim_data)
        pred = sim_model.predict([[len(data) + n_years * 12]])
        preds.append(pred[0])

    return preds

def damage_function(sea_level_rise):
    damage = sea_level_rise ** 3
    return damage

def main():
    file_name = 'GMSL.csv'
    target_year = 2050

    data = read_data(file_name)
    X = np.array(data.index).reshape(-1, 1)
    y = data['GMSL']
    model = fit_linear_regression(X, y)
    
    last_year = pd.to_datetime(data['date'].iloc[-1]).year
    years = np.arange(last_year + 1, target_year + 1)
    damages = []

    for year in tqdm(years, desc='Calculating damages'):
        preds = monte_carlo_projection(model, year, data)
        mean_pred = np.mean(preds)
        damage = damage_function(mean_pred)
        damages.append(damage)

    def y_fmt(x, _):
        return f'{x:.0f}'

    y_formatter = FuncFormatter(y_fmt)

    # Plot the sea level rise projections
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    preds = monte_carlo_projection(model, target_year, data)
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    plt.hist(preds, bins=30)
    plt.xlabel('Sea Level Rise (mm)')
    plt.ylabel('Frequency')
    plt.title(f'Sea Level Rise Projections for {target_year}')

    # Plot the damage estimates over time
    plt.subplot(1, 2, 2)
    plt.plot(years, damages, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Estimated Damage')
    plt.gca().yaxis.set_major_formatter(y_formatter)  # Apply the custom formatter to the y-axis
    plt.title('Damage Estimates Over Time')
    
    plt.tight_layout()
    plt.show()

    print(f"Estimated sea level rise in {target_year}: {mean_pred:.2f} mm (±{std_pred:.2f} mm)")
    print(f"Estimated damage in {target_year}: {damages[-1]:.2e}")

if __name__ == "__main__":
    main()