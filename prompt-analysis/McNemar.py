import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, binom, binomtest

MODELOS = ['gemma2_27b','llama3_1_8b','phi3_5_3_8b','phi4_14b','qwen2_7b'] 
CATEGORIAS = ['day_match', 'hour_match']
alpha = 0.05
crit = chi2.ppf(1 - alpha, 1)
os.makedirs('output/mcnemar', exist_ok=True)

for MODELO in MODELOS:
    for CATEGORIA in CATEGORIAS: 
        df = pd.read_csv('aciertos.csv')
        df = df[df.modelo == MODELO]
        df['x'] = df[CATEGORIA] & ~(df.estrategia.eq('chain_of_thought') & df.fuga_razonamiento)
        X = df.pivot_table(index=['escenario', 'task_id'], columns='estrategia', values='x').astype(bool)

        def mcnemar(j, jp):
            n12 = int((X[j] & ~X[jp]).sum())
            n21 = int((~X[j] & X[jp]).sum())
            ns = n12 + n21
            if ns == 0:
                return 0.0, 0.0, 1.0, n21, ns
            z = (n21 - n12) / np.sqrt(ns)
            p = binomtest(n21, ns, 0.5).pvalue if ns <= 10 else chi2.sf(z**2, 1)
            return z, (n21 - n12) / len(X), p, n21, ns

        res = [[('zero_shot', s, *mcnemar('zero_shot', s)) for s in ['few_shot', 'chaining', 'chain_of_thought']]]
        vivos = [jp for j, jp, z, d, p, n21, ns in res[0] if p < alpha and z > 0] or ['zero_shot']
        if len(vivos) > 1:
            res.append([(j, jp, *mcnemar(j, jp)) for j, jp in itertools.combinations(vivos, 2)])
            perdedores = {(j if z > 0 else jp) for j, jp, z, d, p, n21, ns in res[1] if p < alpha}
            vivos = [s for s in vivos if s not in perdedores]
        print(MODELO, CATEGORIA, 'Ganador(es):', vivos)

        ncol = max(len(r) for r in res)
        fig, axes = plt.subplots(len(res), ncol, figsize=(5 * ncol, 4 * len(res)), squeeze=False)
        for i, fila in enumerate(res):
            for k, (j, jp, z, d, p, n21, ns) in enumerate(fila):
                ax = axes[i, k]
                if ns <= 10:
                    ks = np.arange(ns + 1)
                    pmf = binom.pmf(ks, ns, 0.5)
                    extremo = np.abs(ks - ns / 2) >= abs(n21 - ns / 2)
                    ax.bar(ks[~extremo], pmf[~extremo], color='gray')
                    ax.bar(ks[extremo], pmf[extremo], color='green', alpha=0.5, label=f'p-value = {p:.3g}')
                    rech = ks[binom.cdf(ks, ns, 0.5) <= alpha / 2]
                    if len(rech):
                        ax.axvline(rech.max(), color='r', ls='--', label=f'crítico α={alpha}')
                        ax.axvline(ns - rech.max(), color='r', ls='--')
                    ax.axvline(n21, color='green', label=f'$n_{{21}}$ = {n21} de $n^*$ = {ns}, d = {d:+.3f}')
                    ax.set_ylabel(f'P — Binomial($n^*$={ns}, 1/2)')
                else:
                    T = z**2
                    x = np.linspace(0.01, max(T, crit) * 1.2, 500)
                    ax.plot(x, chi2.pdf(x, 1), 'k')
                    ax.axvline(crit, color='r', ls='--', label=f'crítico α={alpha} ({crit:.2f})')
                    ax.fill_between(x, chi2.pdf(x, 1), where=x >= T, color='green', alpha=0.5, label=f'p-value = {p:.3g}')
                    ax.axvline(T, color='green', label=f'$z_0^2$ = {T:.2f}, d = {d:+.3f}')
                    ax.set_ylabel('Densidad — $\\chi^2_1$')
                ax.set_title(f'Ronda {i+1}: {j} vs {jp}')
                ax.legend()
            for k in range(len(fila), ncol):
                axes[i, k].axis('off')
        fig.suptitle(f'McNemar — {MODELO} — {CATEGORIA}')
        plt.tight_layout()
        plt.savefig(f'output/mcnemar/mcnemar_{MODELO}_{CATEGORIA}.png', dpi=300, bbox_inches='tight')