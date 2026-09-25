import pandas as pd
df = pd.read_csv('./aciertos.csv')

print(df.groupby('estrategia')[['day_match','hour_match','sensor_match','priority_match']].mean().round(3))

cot = df[df.estrategia == 'chain_of_thought']
print('\nFugas CoT por modelo:')
print(cot.groupby('modelo').fuga_razonamiento.agg(['sum','count']))

print('\nDistribución de expected_hour:')
print(df[df.estrategia == 'zero_shot'].expected_hour.value_counts(normalize=True).round(3))