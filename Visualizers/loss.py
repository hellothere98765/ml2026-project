import matplotlib.pyplot as plt

k = [i for i in range(1, 11)]
train_loss = [
    6.8442,
    5.4137,
    4.6837,
    4.1377,
    3.6721,
    3.2817,
    2.9596,
    2.7052,
    2.5074,
    2.3467
]
test_loss = [
    6.7360,
    6.3364,
    6.4238,
    6.5869,
    6.7985,
    6.9911,
    7.1916,
    7.3347,
    7.3811,
    7.4199
]

plt.figure(figsize=(8,5))

plt.plot(k, train_loss, label="Train Loss")
plt.plot(k, test_loss, label="Test Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Test Loss")
plt.legend()
plt.grid(True)

plt.savefig("loss_plot.png", dpi=300, bbox_inches="tight")  # saves image