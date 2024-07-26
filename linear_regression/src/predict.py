import sys

try:
        
    with open("result.txt", "r") as fresult:
        result = fresult.read().split(", ")

        t0, t1, xmax, ymax = float(result[0]), float(result[1]), float(result[2]), float(result[3])

        if len(sys.argv) != 2:
            print("Invalid input.")
            exit(1)

        x = float(sys.argv[1]) / xmax
        y_predict = round(t0 + t1 * x)

        print(f"Predicted price is {y_predict} euros for {sys.argv[1]} km")
except:
    print("Untrained model, cannot predict")