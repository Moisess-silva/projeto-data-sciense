from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
df.to_csv('wine_raw.csv', index=False)
print("Ficheiro wine_raw.csv gerado com sucesso!")