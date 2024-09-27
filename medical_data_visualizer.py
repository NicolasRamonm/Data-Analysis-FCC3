import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Leitura dos dados
df = pd.read_csv('medical_examination.csv')

# 2. Criação da coluna 'overweight' (acima do peso) com base no IMC (Índice de Massa Corporal)
# IMC > 25 é considerado acima do peso. O resultado é convertido para valores binários (1 = acima do peso, 0 = não acima do peso).
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2 > 25).astype(int)

# 3. Normalização dos valores de 'gluc' (glicose) e 'cholesterol' (colesterol)
# Valores > 1 são considerados elevados e são convertidos para 1, enquanto os valores normais são 0.
df[['gluc', 'cholesterol']] = (df[['gluc', 'cholesterol']] > 1).astype(int)

# 4. Função para gerar o gráfico categórico
def draw_cat_plot():
    # 5. Transformação dos dados no formato "long" para facilitar a criação de gráficos categóricos
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6. Criação de gráfico categórico com contagem de variáveis
    # O gráfico mostra a contagem das variáveis (ativas, alcoólatras, colesterol, etc.) para pessoas com e sem problemas cardíacos (cardio).
    fig = sns.catplot(data=df_cat, kind='count', x='variable', hue='value', col='cardio').set(ylabel='total').fig

    # 9. Salva o gráfico categórico em um arquivo
    fig.savefig('catplot.png')
    
    # 10. Retorna o objeto da figura para exibição posterior
    return fig

# 11. Função para gerar o mapa de calor
def draw_heat_map():
    # 12. Limpeza dos dados:
    # Remoção de registros com pressão diastólica maior que a sistólica,
    # e remoção de outliers (2,5% menores e 2,5% maiores) de altura e peso.
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) & 
        (df['height'] <= df['height'].quantile(0.975)) & 
        (df['weight'] >= df['weight'].quantile(0.025)) & 
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 13. Cálculo da matriz de correlação entre as variáveis numéricas
    corr = df_heat.corr()

    # 14. Geração de uma máscara para a parte superior do triângulo da matriz de correlação, evitando duplicação de valores no gráfico
    mask = np.triu(corr)

    # 15. Configuração da figura do matplotlib
    fig, ax = plt.subplots()

    # 16. Criação do mapa de calor usando a matriz de correlação, com anotação dos valores
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='0.1f', square=True)

    # 17. Salvamento do mapa de calor em um arquivo
    fig.savefig('heatmap.png')
    
    # 18. Retorna o objeto da figura para exibição posterior
    return fig
