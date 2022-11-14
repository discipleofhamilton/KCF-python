import cv2
import os
import numpy as np
import time
import numba as nb
import sys
from sklearn.neighbors import KDTree


# Salient color names
color_names = np.array([[255,0,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [255,0,255],
                        [128,0,0], [128,128,0], [0,128,0], [0,128,128], [0,0,128], [128,0,128],
                        [0,0,0], [128,128,128], [192,192,192], [255,255,255]], dtype=np.float32) / 255

colornames_tree = KDTree(color_names, leaf_size=2)


@nb.jit(nopython=True)
def get_Euclideandist(point1: np.ndarray, point2:np.ndarray) -> float:

    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


@nb.jit(nopython=True)
def find_similiar_colorname(cell: np.ndarray) -> np.ndarray:

    global color_names
    # global colornames_tree

    # Brute-force
    min_dist = sys.maxsize
    index    = -1

    # Get mean color of the cell
    mean_cell = np.zeros(3, dtype=np.float32)
    mean_cell[0] = np.mean(cell[:,:,0])
    mean_cell[1] = np.mean(cell[:,:,1])
    mean_cell[2] = np.mean(cell[:,:,2])

    for i in range(len(color_names)):

        # Get the distance of the salient color name and cell
        # New optimaztion for using numba optimizer
        dist = get_Euclideandist(color_names[i], mean_cell)

        if min_dist > dist:
            min_dist = dist
            index    = i

    return color_names[index]

@nb.jit(nopython=True)
def get_mask(output_size: np.ndarray, image: np.ndarray):

    '''
    The function is to split an image into small cells.

    '''

    out_h, out_w = output_size # number of cells on image height, number of cells on image width
    h, w, c      = image.shape

    mask = np.zeros((out_h, out_w, c))

    cell_h = h // out_h
    cell_w = w // out_w

    # Split the cells

    cell = None
    for ch in range(cell_h, h+1, cell_h):
        for cw in range(cell_w, w+1, cell_w):

            cell = image[ch-cell_h:ch, cw-cell_w:cw,:]
            mask[(ch-cell_h)/cell_h:ch/cell_h,(cw-cell_w)/cell_w:cw/cell_w,:] = find_similiar_colorname(cell=cell)

    return mask


def connect_background(mask: np.ndarray) -> np.ndarray:

    h, w, c = mask.shape


def show_colornames(cn_unit: int = 100):

    cn = np.zeros((cn_unit, 16*cn_unit, 3))

    for i in range(color_names.shape[0]):
        cn[:,i*cn_unit:(i+1)*cn_unit,:] = color_names[i]

    cv2.imshow("color names", cn)


if __name__ == '__main__':

    # Show color names
    # show_colornames()

    '''
    Version conflict: the original camera capture is not working
    cap  = cv2.VideoCapture(0)
    '''

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 8, 6
    size = np.array([40 ,30])
    # size = np.array([32, 24])
    # size = np.array([24, 18])
    # size = np.array([16 ,12])

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    total_exe_time = 0
    frame_counter  = 0

    while True:

        # Get frame
        ret, frame = cap.read()

        if not ret:
            print('Can not receive frame!!!')
            break

        frame_counter += 1

        # normalize the image
        frame_norm = frame / 255

        st_getmask = time.time()

        # Get mask
        mask = get_mask(output_size=size, image=frame_norm)
        # print(mask.shape)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

        end_getmask = time.time()
        exe_time = end_getmask-st_getmask

        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)

        if frame_counter > 1:
            total_exe_time += exe_time

        print('get mask time: %.3fms' % (exe_time*1000))

        # Processin with mask
        res = cv2.bitwise_and(mask, frame_norm)

        # Show image
        # cv2.imshow('video', frame)
        concate = np.concatenate((frame_norm, mask, res), axis=1)
        cv2.imshow('res', concate)

        # Press 'q', 'ESC', 'SPACE' to exit the iteration
        break_point = cv2.waitKey(1)
        if break_point == ord('q') or \
           break_point == 27 or \
           break_point == 32:
            break

    cap.release()
    cv2.destroyAllWindows()

    print('\ncell size: %s, mean exe time: %.3fms' % (str(size), total_exe_time*1000/(frame_counter-1)))