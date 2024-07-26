import matplotlib.pyplot as plt
from utils import get_courses_data, open_arg_csv, get_color_per_house
from math import ceil

def main():
    pcsv = open_arg_csv()

    houses = dict.fromkeys(set(d for d in pcsv.iloc[:, 1]))
    courses_names = pcsv.columns[6:].array
    col_number = len(courses_names)

    fig, ax = plt.subplots(ceil(col_number / 3), 3, figsize=(22, 16))
    fig.tight_layout()

    i = 0
    for course in courses_names:
        sx = int(i / 3)
        sy = i % 3
        create_histogram_for_course(pcsv, ax, sx, sy, houses, course)
        i = i + 1

    for j in range((ceil(col_number / 3) * 3) - i):
        ax[int((j + i) / 3), (j + i) % 3].axis("off")

    plt.show()

def create_histogram_for_course(csv, plot, px,  py, houses, course):
    for house in houses:
        house_data = csv[csv['Hogwarts House'] == house]
        course_data = house_data[[course]]
        plot[px, py].hist(course_data, bins=42, alpha=0.5, color=get_color_per_house(house), label=house)

if __name__ == "__main__":
    main()
