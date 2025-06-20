import numpy as np

with open("Q.npy", "rb") as f:
    a = np.load(f)
    print(a)
