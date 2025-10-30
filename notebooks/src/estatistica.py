"""Funções auxiliares para análises estatísticas.

Inclui testes de Levene, t de Student (independente), Mann-Whitney
e utilitário para remoção de outliers.
"""

from scipy.stats import (
    levene,
    mannwhitneyu,
    ttest_ind,
)


def analise_levene(dataframe, alfa=0.05, centro="mean"):
    """Executa o teste de Levene para igualdade de variâncias.

    Args:
        dataframe: DataFrame com duas ou mais colunas numéricas a comparar.
        alfa: Nível de significância para a decisão.
        centro: Estatística de centralidade usada pelo teste ("mean" ou "median").
    """
    print("Teste de Levene")

    estatistica_levene, valor_p_levene = levene(
        *[dataframe[coluna] for coluna in dataframe.columns],
        center=centro,
        nan_policy="omit",
    )

    print(f"{estatistica_levene=:.3f}")
    if valor_p_levene > alfa:
        print(f"Variâncias iguais (valor p: {valor_p_levene:.3f})")
    else:
        print(f"Ao menos uma variância é diferente (valor p: {valor_p_levene:.3f})")


def analise_ttest_ind(
    dataframe,
    alfa=0.05,
    variancias_iguais=True,
    alternativa="two-sided",
):
    """Executa o teste t de Student para amostras independentes.

    Args:
        dataframe: DataFrame com duas colunas numéricas a comparar.
        alfa: Nível de significância para a decisão.
        variancias_iguais: Se True, assume variâncias iguais (teste padrão).
        alternativa: Tipo de hipótese alternativa ("two-sided", "less", "greater").
    """
    print("Teste t de Student")
    estatistica_ttest, valor_p_ttest = ttest_ind(
        *[dataframe[coluna] for coluna in dataframe.columns],
        equal_var=variancias_iguais,
        alternative=alternativa,
        nan_policy="omit",
    )

    print(f"{estatistica_ttest=:.3f}")
    if valor_p_ttest > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_ttest:.3f})")


def analise_mannwhitneyu(
    dataframe,
    alfa=0.05,
    alternativa="two-sided",
):
    """Executa o teste de Mann-Whitney para duas amostras independentes.

    Args:
        dataframe: DataFrame com duas colunas numéricas a comparar.
        alfa: Nível de significância para a decisão.
        alternativa: Tipo de hipótese alternativa ("two-sided", "less", "greater").
    """

    print("Teste de Mann-Whitney")
    estatistica_mw, valor_p_mw = mannwhitneyu(
        *[dataframe[coluna] for coluna in dataframe.columns],
        nan_policy="omit",
        alternative=alternativa,
    )

    print(f"{estatistica_mw=:.3f}")
    if valor_p_mw > alfa:
        print(f"Não rejeita a hipótese nula (valor p: {valor_p_mw:.3f})")
    else:
        print(f"Rejeita a hipótese nula (valor p: {valor_p_mw:.3f})")


def remove_outliers(dados, largura_bigodes=1.5):
    """Remove outliers via regra do IQR (bigodes de boxplot).

    Args:
        dados: Série pandas com valores numéricos.
        largura_bigodes: Multiplicador do IQR para limites inferior e superior.
    """
    q1 = dados.quantile(0.25)
    q3 = dados.quantile(0.75)
    iqr = q3 - q1
    return dados[(dados >= q1 - largura_bigodes * iqr) & (dados <= q3 + largura_bigodes * iqr)]
