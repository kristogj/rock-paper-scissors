import matplotlib.pyplot as plt


def generate_lineplot(losses):
    epochs = list(range(1, len(losses) + 1))
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epochs, losses)
    plt.savefig("./graphs/loss.png")
    plt.show()
