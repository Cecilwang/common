import numpy as np


def main():
    print("Hello World!")
    print(np.arange(4).reshape(2, 2))
    print(np.unravel_index(3, (2, 2)))


if __name__ == "__main__":
    main()
