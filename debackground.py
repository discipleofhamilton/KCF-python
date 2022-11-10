from re import I
import cv2
import os
import numpy as np
import time
import numba as nb


# SIZE = np.array([16 ,12])
# # SIZE = np.array([8 ,6])

# # Salient color names
# COLOR_NAMES = np.array([[255,0,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [255,0,255],
#                         [128,0,0], [128,128,0], [0,128,0], [0,128,128], [0,0,128], [128,0,128],
#                         [0,0,0], [128,128,128], [192,192,192], [255,255,255]]) / 255


@nb.njit
def find_similiar_colorname(cell: np.ndarray) -> np.ndarray:

    # Salient color names
    color_names = np.array([[255,0,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [255,0,255],
                            [128,0,0], [128,128,0], [0,128,0], [0,128,128], [0,0,128], [128,0,128],
                            [0,0,0], [128,128,128], [192,192,192], [255,255,255]]) / 255

    # Brute-force
    max_dist = 0
    index    = -1

    for i in range(len(color_names)):

        # Get mean color of the cell
        mean_cell = np.mean(cell, axis=0)

        # Get the distance of the salient color name and cell
        dist = np.linalg.norm(color_names[i] - mean_cell)

        if max_dist < dist:
            max_dist = dist
            index    = i

    # # kd tree
    # from sklearn.neighbors import KDTree

    return color_names[index]

@nb.njit
def get_mask(output_size: np.ndarray, image: np.ndarray):

    '''
    The function is to split an image into small cells.

    '''

    out_h, out_w = output_size # number of cells on image height, number of cells on image width
    h, w, c      = image.shape

    mask = np.zeros((h,w,c))

    cell_h = h // out_h
    cell_w = w // out_w

    st = time.time()
    # Split the cells

    cell = None
    for ch in range(cell_h, h+1, cell_h):
        for cw in range(cell_w, w+1, cell_w):

            cell = image[ch-cell_h:ch, cw-cell_w:cw,:].copy()
            # st = time.time()
            mask[ch-cell_h:ch, cw-cell_w:cw,:] = find_similiar_colorname(cell=cell)


    end = time.time()

    print('get each similar color name time: %.3fms' % ((end-st)*1000))

    return mask


def connect_background(mask: np.ndarray) -> np.ndarray:

    h, w, c = mask.shape


if __name__ == '__main__':

    cap  = cv2.VideoCapture(0)
    size = np.array([16 ,12])

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:

        # Get frame
        ret, frame = cap.read()

        if not ret:
            print('Can not receive frame!!!')
            break

        # normalize the image
        frame_norm = frame / 255

        # Get mask
        mask = get_mask(output_size=size, image=frame_norm)
        # print(mask.shape)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)

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