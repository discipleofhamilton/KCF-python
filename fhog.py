import cv2
import numba as nb
import numpy as np

# constant
NUM_SECTOR = 9
FLT_EPSILON = 1e-07


@nb.jit(nopython=True)
def func1(dx, dy, boundary_x, boundary_y, height, width, num_channels):
    r = np.zeros((height, width), dtype="float32")
    alfa = np.zeros((height, width, 2), dtype="int")

    for j in range(1, height - 1):
        for i in range(1, width - 1):
            c = 0
            x = dx[j, i, c]
            y = dy[j, i, c]
            r[j, i] = np.sqrt(x * x + y * y)

            for ch in range(1, num_channels):
                tx = dx[j, i, ch]
                ty = dy[j, i, ch]
                magnitude = np.sqrt(tx * tx + ty * ty)
                if magnitude > r[j, i]:
                    r[j, i] = magnitude
                    c = ch
                    x = tx
                    y = ty

            mmax = boundary_x[0] * x + boundary_y[0] * y
            maxi = 0

            for kk in range(0, NUM_SECTOR):
                dot_prod = boundary_x[kk] * x + boundary_y[kk] * y
                if dot_prod > mmax:
                    mmax = dot_prod
                    maxi = kk
                elif -dot_prod > mmax:
                    mmax = -dot_prod
                    maxi = kk + NUM_SECTOR

            alfa[j, i, 0] = maxi % NUM_SECTOR
            alfa[j, i, 1] = maxi
    return r, alfa.astype("uint8")


@nb.jit(nopython=True)
def func2(dx, dy, boundary_x, boundary_y, r, alfa, nearest, w, k, height, width, size_x, size_y, p, string_size):
    mapp = np.zeros((size_x * size_y * p), dtype=np.float32)

    for i in range(size_y):
        for j in range(size_x):
            for ii in range(k):
                for jj in range(k):
                    if (i * k + ii > 0) and (i * k + ii < height - 1) and (j * k + jj > 0) and (j * k + jj < width - 1):
                        mapp[i * string_size + j * p + alfa[k * i + ii, j * k + jj, 0]] += (
                            r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]
                        )
                        mapp[i * string_size + j * p + alfa[k * i + ii, j * k + jj, 1] + NUM_SECTOR] += (
                            r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 0]
                        )
                        if (i + nearest[ii] >= 0) and (i + nearest[ii] <= size_y - 1):
                            mapp[(i + nearest[ii]) * string_size + j * p + alfa[k * i + ii, j * k + jj, 0]] += (
                                r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 0]
                            )
                            mapp[
                                (i + nearest[ii]) * string_size + j * p + alfa[k * i + ii, j * k + jj, 1] + NUM_SECTOR
                            ] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 0]
                        if (j + nearest[jj] >= 0) and (j + nearest[jj] <= size_x - 1):
                            mapp[i * string_size + (j + nearest[jj]) * p + alfa[k * i + ii, j * k + jj, 0]] += (
                                r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 1]
                            )
                            mapp[
                                i * string_size + (j + nearest[jj]) * p + alfa[k * i + ii, j * k + jj, 1] + NUM_SECTOR
                            ] += r[k * i + ii, j * k + jj] * w[ii, 0] * w[jj, 1]
                        if (
                            (i + nearest[ii] >= 0)
                            and (i + nearest[ii] <= size_y - 1)
                            and (j + nearest[jj] >= 0)
                            and (j + nearest[jj] <= size_x - 1)
                        ):
                            mapp[
                                (i + nearest[ii]) * string_size
                                + (j + nearest[jj]) * p
                                + alfa[k * i + ii, j * k + jj, 0]
                            ] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]
                            mapp[
                                (i + nearest[ii]) * string_size
                                + (j + nearest[jj]) * p
                                + alfa[k * i + ii, j * k + jj, 1]
                                + NUM_SECTOR
                            ] += r[k * i + ii, j * k + jj] * w[ii, 1] * w[jj, 1]
    return mapp


@nb.jit(nopython=True)
def func3(part_of_norm, mappmap, size_x, size_y, p, xp, pp):
    new_data = np.zeros((size_y * size_x * pp), np.float32)
    for i in range(1, size_y + 1):
        for j in range(1, size_x + 1):
            pos1 = i * (size_x + 2) * xp + j * xp
            pos2 = (i - 1) * size_x * pp + (j - 1) * pp

            val_of_norm = (
                np.sqrt(
                    part_of_norm[(i) * (size_x + 2) + (j)]
                    + part_of_norm[(i) * (size_x + 2) + (j + 1)]
                    + part_of_norm[(i + 1) * (size_x + 2) + (j)]
                    + part_of_norm[(i + 1) * (size_x + 2) + (j + 1)]
                )
                + FLT_EPSILON
            )
            new_data[pos2 : pos2 + p] = mappmap[pos1 : pos1 + p] / val_of_norm
            new_data[pos2 + 4 * p : pos2 + 6 * p] = mappmap[pos1 + p : pos1 + 3 * p] / val_of_norm

            val_of_norm = (
                np.sqrt(
                    part_of_norm[(i) * (size_x + 2) + (j)]
                    + part_of_norm[(i) * (size_x + 2) + (j + 1)]
                    + part_of_norm[(i - 1) * (size_x + 2) + (j)]
                    + part_of_norm[(i - 1) * (size_x + 2) + (j + 1)]
                )
                + FLT_EPSILON
            )
            new_data[pos2 + p : pos2 + 2 * p] = mappmap[pos1 : pos1 + p] / val_of_norm
            new_data[pos2 + 6 * p : pos2 + 8 * p] = mappmap[pos1 + p : pos1 + 3 * p] / val_of_norm

            val_of_norm = (
                np.sqrt(
                    part_of_norm[(i) * (size_x + 2) + (j)]
                    + part_of_norm[(i) * (size_x + 2) + (j - 1)]
                    + part_of_norm[(i + 1) * (size_x + 2) + (j)]
                    + part_of_norm[(i + 1) * (size_x + 2) + (j - 1)]
                )
                + FLT_EPSILON
            )
            new_data[pos2 + 2 * p : pos2 + 3 * p] = mappmap[pos1 : pos1 + p] / val_of_norm
            new_data[pos2 + 8 * p : pos2 + 10 * p] = mappmap[pos1 + p : pos1 + 3 * p] / val_of_norm

            val_of_norm = (
                np.sqrt(
                    part_of_norm[(i) * (size_x + 2) + (j)]
                    + part_of_norm[(i) * (size_x + 2) + (j - 1)]
                    + part_of_norm[(i - 1) * (size_x + 2) + (j)]
                    + part_of_norm[(i - 1) * (size_x + 2) + (j - 1)]
                )
                + FLT_EPSILON
            )
            new_data[pos2 + 3 * p : pos2 + 4 * p] = mappmap[pos1 : pos1 + p] / val_of_norm
            new_data[pos2 + 10 * p : pos2 + 12 * p] = mappmap[pos1 + p : pos1 + 3 * p] / val_of_norm
    return new_data


@nb.jit(nopython=True)
def func4(mappmap, p, size_x, size_y, pp, yp, xp, nx, ny):
    new_data = np.zeros((size_x * size_y * pp), np.float32)
    for i in range(size_y):
        for j in range(size_x):
            pos1 = (i * size_x + j) * p
            pos2 = (i * size_x + j) * pp

            for jj in range(2 * xp):  # 2*9
                new_data[pos2 + jj] = np.sum(mappmap[pos1 + yp * xp + jj : pos1 + 3 * yp * xp + jj : 2 * xp]) * ny
            for jj in range(xp):  # 9
                new_data[pos2 + 2 * xp + jj] = np.sum(mappmap[pos1 + jj : pos1 + jj + yp * xp : xp]) * ny
            for ii in range(yp):  # 4
                new_data[pos2 + 3 * xp + ii] = (
                    np.sum(mappmap[pos1 + yp * xp + ii * xp * 2 : pos1 + yp * xp + ii * xp * 2 + 2 * xp]) * nx
                )
    return new_data


def get_feature_maps(image, k, mapp):
    kernel = np.array([[-1.0, 0.0, 1.0]], np.float32)

    height = image.shape[0]
    width = image.shape[1]
    assert image.ndim == 3 and image.shape[2]
    num_channels = 3  # (1 if image.ndim==2 else image.shape[2])

    size_x = width // k
    size_y = height // k
    px = 3 * NUM_SECTOR
    p = px
    string_size = size_x * p

    mapp["size_x"] = size_x
    mapp["size_y"] = size_y
    mapp["num_features"] = p
    mapp["map"] = np.zeros(int(mapp["size_x"] * mapp["size_y"] * mapp["num_features"]), dtype=np.float32)

    dx = cv2.filter2D(np.float32(image), -1, kernel)  # np.float32(...) is necessary
    dy = cv2.filter2D(np.float32(image), -1, kernel.T)

    arg_vector = np.arange(NUM_SECTOR + 1).astype(np.float32) * np.pi / NUM_SECTOR
    boundary_x = np.cos(arg_vector)
    boundary_y = np.sin(arg_vector)

    r, alfa = func1(dx, dy, boundary_x, boundary_y, height, width, num_channels)  # with @jit

    nearest = np.ones((k), np.int64)
    nearest[0 : k // 2] = -1

    w = np.zeros((k, 2), np.float32)
    a_x = np.concatenate((k / 2 - np.arange(k // 2) - 0.5, np.arange(k // 2, k) - k // 2 + 0.5)).astype(np.float32)
    b_x = np.concatenate((k / 2 + np.arange(k // 2) + 0.5, -np.arange(k // 2, k) + k // 2 - 0.5 + k)).astype(np.float32)
    w[:, 0] = 1.0 / a_x * ((a_x * b_x) / (a_x + b_x))
    w[:, 1] = 1.0 / b_x * ((a_x * b_x) / (a_x + b_x))

    mapp["map"] = func2(
        dx.astype(np.int8),
        dy.astype(np.int8),
        boundary_x,
        boundary_y,
        r,
        alfa,
        nearest,
        w,
        k,
        height,
        width,
        int(size_x),
        int(size_y),
        p,
        int(string_size),
    )  # with @jit

    return mapp


def normalize_and_truncate(mapp, alfa):
    size_x = int(mapp["size_x"])
    size_y = int(mapp["size_y"])

    p = NUM_SECTOR
    xp = NUM_SECTOR * 3
    pp = NUM_SECTOR * 12

    idx = np.arange(0, size_x * size_y * mapp["num_features"], mapp["num_features"]).reshape(
        (size_x * size_y, 1)
    ) + np.arange(p)
    part_of_norm = np.sum(mapp["map"][idx] ** 2, axis=1)  ### ~0.0002s

    size_x, size_y = size_x - 2, size_y - 2

    new_data = func3(part_of_norm, mapp["map"], size_x, size_y, p, xp, pp)  # with @jit

    # truncation
    new_data[new_data > alfa] = alfa

    mapp["num_features"] = pp
    mapp["size_x"] = size_x
    mapp["size_y"] = size_y
    mapp["map"] = new_data

    return mapp


def pca_feature_maps(mapp):
    size_x = mapp["size_x"]
    size_y = mapp["size_y"]

    p = mapp["num_features"]
    pp = NUM_SECTOR * 3 + 4
    yp = 4
    xp = NUM_SECTOR

    nx = 1.0 / np.sqrt(xp * 2)
    ny = 1.0 / np.sqrt(yp)

    new_data = func4(mapp["map"], p, size_x, size_y, pp, yp, xp, nx, ny)  # with @jit

    mapp["num_features"] = pp
    mapp["map"] = new_data

    return mapp
