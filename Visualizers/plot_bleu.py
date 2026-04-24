import matplotlib.pyplot as plt

models = list(range(1, 11))
bleu_scores = [
    0.01,
    0.09,
    0.14,
    0.09,
    0.12,
    0.07,
    0.09,
    0.08,
    0.10,
    0.07
    ]
bleu_scores = [100*i for i in bleu_scores]

plt.figure(figsize=(8, 5))
plt.plot(models, bleu_scores, marker='o')
plt.title("BLEU Score Over Models")
plt.xlabel("Model Number")
plt.ylabel("BLEU Score")
plt.xticks(models)
plt.grid(True)
plt.savefig("bleu_scores.png")