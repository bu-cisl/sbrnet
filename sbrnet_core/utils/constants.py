import numpy as np
from typing import List

CM2_SIZE = [2076, 3088]
NUM_VIEWS = 9
FOCUS_LOC = np.array(
    [
        [406, 909],
        [407, 1545],
        [405, 2175],
        [1037, 911],
        [1037, 1544],
        [1037, 2176],
        [1675, 911],
        [1675, 1543],
        [1675, 2173],
    ]
)
NUM_SLICES = 24  # number of slices in the refocus volume

# 0 | 1 | 2
# ---------
# 3 | 4 | 5
# ---------
# 6 | 7 | 8
view_combos = [
    [4],
    [4, 5],
    [0, 4],
    [0, 1, 4],
    [0, 4, 8],
    [1, 3, 5, 7],
    [0, 2, 6, 8],
    [1, 3, 4, 5, 7],
    [0, 2, 4, 6, 8],
    [0, 2, 4, 5, 6, 8],
    [0, 2, 3, 4, 5, 6, 8],
    [0, 1, 2, 3, 5, 6, 7, 8],  # no center
    [0, 1, 3, 4, 5, 6, 7, 8],  # no top right (aberrated)
    [0, 2, 3, 4, 5, 6, 7, 8],  # no top middle
    [0, 1, 2, 3, 4, 5, 6, 7, 8], # all
]
