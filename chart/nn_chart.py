import matplotlib.pyplot as plt
from numpy import sqrt, floor, ceil, ones, reshape

class NNChart():

    def __init__(self):
        pass

    def display(self, x, width=None, order='F', enable_max_val=False):
        if width is None:
            width = round(sqrt(x.shape[1]))

        m, n = x.shape
        height = n / width
        display_rows = int(floor(sqrt(m)))
        display_cols = int(ceil(m / display_rows))
        pad = 1
        width = int(width)
        height = int(height)
        display_array = ones((pad + display_rows * (height + pad),
                               pad + display_cols * (width + pad)))
        curr_ex = 0
        for i in range(display_rows):
            for j in range(display_cols):
                if curr_ex > m:
                    break
                if enable_max_val:
                    max_val = max(abs(x[curr_ex,:]))
                else:
                    max_val = 1

                start_row = (pad*(i+1) + i * height)
                stop_row = start_row + height
                start_col = (pad*(j+1) + j*width)
                stop_col = start_col + width
                display_array[start_row: stop_row, start_col:stop_col] \
                    = reshape(x[curr_ex, :], (height, width), order=order) / max_val
                curr_ex += 1
            if curr_ex > m:
                break
        plt.imshow(display_array, cmap="gray")
        plt.show()

