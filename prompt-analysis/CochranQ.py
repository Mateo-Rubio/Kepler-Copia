import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

df = pd.read_csv('aciertos.csv')
fuga = df.estrategia.eq('chain_of_thought') & df.fuga_razonamiento
cats = ['day_match', 'hour_match']
alpha, c = 0.05, 4
k = c - 1
crit = chi2.ppf(1 - alpha, k)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for cat, ax in zip(cats, axes.flat):
    df['x'] = (df[cat] & ~fuga).astype(int)
    X = df.pivot_table(index=['modelo', 'escenario', 'task_id'], columns='estrategia', values='x')
    C, R = X.sum(axis=0), X.sum(axis=1)
    N = C.sum()
    T = c * (c - 1) * ((C - N / c) ** 2).sum() / (R * (c - R)).sum()
    p = chi2.sf(T, k)
    print(f'{cat}: T = {T:.2f}, p = {p:.3g}')

    x = np.linspace(0, max(crit,T) * 1.1, 500)
    ax.loglog(x, chi2.pdf(x, k), 'k')
    ax.axvline(crit, color='r', ls='--', label=f'crítico α={alpha} ({crit:.2f})')
    ax.fill_between(x, chi2.pdf(x, k), where=x >= T, color='green', alpha=0.5, label=f'p-value = {p:.3g}')
    ax.axvline(T, color='green', label=f'T = {T:.2f}')
    ax.set_title(f'Prueba estadística para {cat}')
    ax.set_ylabel(r'Valor de la distribución ${\chi^2}$')
    ax.legend()

plt.tight_layout()
plt.savefig("CochranQ.png")