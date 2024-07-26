import csv
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
EPOCHS = 3000

def estPrice(t0: float, t1: float, n):
    return t0 + (t1 * n)

def main():
    dataset = []

    with open('data.csv') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        next(csv_reader)
        dataset = [list(map(float, row)) for row in csv_reader]

    dataset = np.array(dataset)
    maxes = [dataset[:, 0].max(), 1]
    dataset = dataset / maxes

    theta0 = 0
    theta1 = 0

    m = 1 / ( len(dataset))
    for epoch in range(EPOCHS):
        tmp0 = 0.0
        tmp1 = 0.0

        for data in dataset:
            estPriceValue = estPrice(theta0, theta1, data[0]) - data[1]
            tmp0 += estPriceValue
            tmp1 += estPriceValue * data[0]

        new_theta0 = theta0 - LEARNING_RATE * m * tmp0
        new_theta1 = theta1 - LEARNING_RATE * m * tmp1

        diff_theta0 = theta0 - new_theta0
        diff_theta1 = theta1 - new_theta1

        theta0 = new_theta0
        theta1 = new_theta1

        if abs(diff_theta0 + diff_theta1) * 0.5 < 0.001:
            print("Early stopping.")
            break

    plt.figure(figsize=(14, 10))

    rss = 0
    tss = 0
    for data in dataset:
        x = data[0]
        y = data[1]
        y_predict = estPrice(theta0, theta1, x)

        rss += (y - y_predict) ** 2
        tss += (y - dataset[:, 1].mean()) ** 2

        plt.plot([x * maxes[0], x * maxes[0]], [y, y_predict], "c--" )

    print(f"Stopped at {epoch} / {EPOCHS} epochs")

    # R-Squared Score (R2) (https://www.freecodecamp.org/news/evaluation-metrics-for-regression-problems-machine-learning/)
    print(f"Training accuracy : {round((1 - (rss / tss)) * 100)} %")

    plt.plot([maxes[0], 0], [theta1 + theta0, theta0], "r-")
    plt.scatter(dataset[:, 0] * maxes[0], dataset[:, 1] * maxes[1], s=80, color = "blue", marker='+')
    plt.show()

    with open('result.txt', 'w') as f:
        f.write(f"{theta0}, {theta1}, {maxes[0]}, {maxes[1]}")

main()