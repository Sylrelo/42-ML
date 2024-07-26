from utils import open_arg_csv
import matplotlib.pyplot as plt
from utils import get_color_per_house
from histogram import create_histogram_for_course

def generate(include_histogram: bool):
    pcsv = open_arg_csv()
    houses = dict.fromkeys(set(d for d in pcsv.iloc[:, 1]))
    courses_names = pcsv.columns[6:].array

    fig, ax = plt.subplots(13, 13, figsize=(16, 10))
    fig.tight_layout()

    for i in range(13):
        col_name = courses_names[i]
        ax[0, i].set_title(col_name)
        ax[i, 0].set_ylabel(col_name, rotation=90, size=10)
        for j in range(13):
            disable_axis_labels(ax[i, j])
            if j == i:
                if include_histogram:
                    create_histogram_for_course(pcsv, ax, i, j, houses, courses_names[i])
                else:
                    ax[i, j].axis("off")
                continue
            for house in houses:
                house_data = pcsv[pcsv['Hogwarts House'] == house].iloc
                ax[i, j].scatter(house_data[:, 6 + j], house_data[:, 6 + i], alpha=.3, color=get_color_per_house(house))
            
    plt.show()

def disable_axis_labels(ax):
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.set_xticks([])
    ax.set_yticks([])

if __name__ == "__main__":
    generate(False)
