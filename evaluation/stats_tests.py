import numpy as np
from scipy.stats import ttest_rel

# Data for Faithfulness (Cosine Similarity)
faithfulness_rag = np.array([0.80, 0.80, 0.83, 0.75, 0.76, 0.74, 0.80, 0.76, 0.81])
faithfulness_non_rag = np.array([0.71, 0.73, 0.76, 0.55, 0.69, 0.65, 0.71, 0.66, 0.68])

# Data for Answer Relevance
relevance_rag = np.array([0.81, 0.84, 0.80, 0.74, 0.81, 0.87, 0.78, 0.89, 0.79])
relevance_non_rag = np.array([0.79, 0.80, 0.83, 0.74, 0.81, 0.84, 0.81, 0.85, 0.80])

# Conduct paired t-test for Faithfulness (Cosine Similarity)
t_stat_faithfulness, p_value_faithfulness = ttest_rel(faithfulness_rag, faithfulness_non_rag)
print(f"Paired t-test for Faithfulness (Cosine Similarity):")
print(f"T-statistic: {t_stat_faithfulness}, p-value: {p_value_faithfulness}")

# Conduct paired t-test for Answer Relevance
t_stat_relevance, p_value_relevance = ttest_rel(relevance_rag, relevance_non_rag)
print(f"\nPaired t-test for Answer Relevance:")
print(f"T-statistic: {t_stat_relevance}, p-value: {p_value_relevance}")
