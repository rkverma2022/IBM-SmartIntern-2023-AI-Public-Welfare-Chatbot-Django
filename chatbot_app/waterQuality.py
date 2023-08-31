import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


# Explicitly set keepdims to avoid the warning


def WQ(message):
    try:
        data = pd.read_csv("water_dataset.csv")
        data = data.drop(['id'], axis=1)
        d = data
        d = d.drop(['Potability'], axis=1)
        maxValueForNormalization = d.max().to_numpy(dtype=float)

        data = data / data.max()

        data = data.to_numpy()
        x = data[:, :-1]
        y = data[:, -1]

        # X = message

        # X = np.array([X], dtype=float) / maxValueForNormalization
        X = message / maxValueForNormalization
        X = X.reshape(1, 9)

        KNN = KNeighborsClassifier()
        KNN.fit(x, y)

        result = KNN.predict(X)

        if result == 0:
            return "Water Parameters Shows that it is not safe for drinking"
        elif result == 1:
            return "Water is safe ! You can drink it "
    except FileNotFoundError:
        return "File not found! Please check the file path."
    except (ValueError, IndexError):
        return "Data format issue or missing parameters."


if __name__ == '__main__':
    ans = WQ([7.3, 175, 9460.3, 5.9, 402.0, 15.6, 37.3, 3.2, 5.0])
    print(ans)
