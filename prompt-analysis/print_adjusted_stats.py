import json
for l in open('analysis/texts.jsonl', encoding='utf-8') if False else []: pass
import pandas as pd
df = pd.read_csv('aciertos.csv')

# 1) CoT con las fugas contadas como fallo, junto a las demás
df['ajustado'] = df.estrategia.eq('chain_of_thought') & df.fuga_razonamiento
for c in ['day_match','hour_match','sensor_match','priority_match']:
    df[c+'_aj'] = df[c] & ~df['ajustado']
print(df.groupby('estrategia')[['day_match_aj','hour_match_aj', 'sensor_match_aj', 'priority_match_aj']].mean().round(3))

# 2) Por modelo, Day y Hour ajustados
print(df.pivot_table(index='modelo', columns='estrategia', values='hour_match_aj').round(3))
print(df.pivot_table(index='modelo', columns='estrategia', values='day_match_aj').round(3))