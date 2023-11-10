import numpy as np
import matplotlib.pyplot as plt


def plot_arrays(x_array: np.ndarray, y_array: np.ndarray, z_array: np.ndarray, clusters: np.ndarray = None,
                figsize: tuple = (10, 10)) -> None:
    fig, ax = plt.subplots(2, figsize=figsize)
    # plt.subplots_adjust(left=0.09, bottom=0.2, right=0.97)
    scatters = [0, 0]
    colors = np.array(["gray", "red", "blue", "yellow", "cyan", 'green', 'black'])

    if clusters is not None:
        c = colors[clusters % (len(colors) - 1)]
        c[clusters == -1] = colors[-1]
    else:
        c = None

    scatters[0] = ax[0].scatter(z_array, x_array, s=10, c=c)
    scatters[1] = ax[1].scatter(z_array, y_array, s=10, c=c)

    # plt.axis([0, 1, -20, 20])

    ax[0].set_ylim(x_array.min(), x_array.max())
    ax[0].grid()

    ax[1].set_ylim(y_array.min(), y_array.max())
    ax[1].grid()
    # ##################################################################3

    plt.show()


if __name__ == "__main__":
    pass
