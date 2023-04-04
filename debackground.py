import cv2
import os
import numpy as np
import time
import numba as nb
import sys
import argparse
from utils import get_error_info


# Salient color names
color_names = np.array([[255,0,0], [255,255,0], [0,255,0], [0,255,255], [0,0,255], [255,0,255],
                        [128,0,0], [128,128,0], [0,128,0], [0,128,128], [0,0,128], [128,0,128],
                        [0,0,0], [128,128,128], [192,192,192], [255,255,255]], dtype=np.float32) / 255


@nb.jit(nopython=True)
def get_Euclideandist(point1: np.ndarray, point2:np.ndarray) -> float:

    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


@nb.jit(nopython=True)
def find_similiar_colorname(cell: np.ndarray) -> np.ndarray:

    global color_names

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
def get_grids(output_size: np.ndarray, image: np.ndarray):

    '''
    The function is to split an image into small grids.
    and get the mask by the color of each grids.
    '''

    out_h, out_w = output_size # number of cells on image height, number of cells on image width
    h, w, c      = image.shape

    cells = np.zeros((out_h, out_w, c))

    cell_h = h // out_h
    cell_w = w // out_w

    # Split the cells

    cell = None
    for ch in range(cell_h, h+1, cell_h):
        for cw in range(cell_w, w+1, cell_w):

            cell = image[ch-cell_h:ch, cw-cell_w:cw,:]
            cells[(ch-cell_h)//cell_h:ch//cell_h,\
                  (cw-cell_w)//cell_w:cw//cell_w,:] = find_similiar_colorname(cell=cell)

    # return connect_background(cells)

    return cells


# @nb.jit(nb.none(nb.float64[:,:,:]), nopython=True)
# @nb.jit(nb.float64[:,:,:](nb.float64[:,:,:]), nopython=True)
@nb.jit(nopython=True)
def connect_background(colornames: np.ndarray) -> np.ndarray:

    h, w, c = colornames.shape

    background = np.zeros((h, w))

    # boundary is background
    background[0, :]   = 1
    background[h-1, :] = 1
    background[:, 0]   = 1
    background[:, w-1] = 1

    '''
    Here is a simple concept:
    1. Get background colors first
    2. Compare with the near 8 bins of its color name. 
       Turn the bins which has same color of the background to background.

    Here comes a flaut/fraction.
    If the bin which has the same color to background may be not found.
    '''

    # To-Do
    '''
    Replace the simple method with connected components algorithm.
    '''

    # filter
    f = np.array([[0,1,0],[1,1,1],[0,1,0]])

    for i in range(1, h-1):
        for j in range(1, w-1):

            # get backgrounds
            valid_bg = np.logical_and(background[i-1:i+2, j-1:j+2], f)
            valid_cn = colornames[i-1:i+2, j-1:j+2]

            for n in range(valid_bg.shape[0]):
                for m in range(valid_bg.shape[1]):
                    if valid_bg[n,m] and \
                       np.array_equal(valid_cn[n,m], colornames[i, j]):
                        background[i, j] = 1 
                        break 

    return 1 - background


@nb.jit(nopython=True)
def bitwise(img: np.ndarray, mask: np.ndarray) -> np.ndarray:

    res = None
    h, w = 0, 0

    if img.ndim == 3:
        h, w, c = img.shape
        res = np.zeros((h, w, c))

    else:
        h, w = img.shape
        res = np.zeros((h, w))
        
    # For numba optimization
    # loop the mask to check which pixels should get through
    for n in range(h):
        for m in range(w):
            if mask[n,m] == 1:
                res[n,m] = img[n,m]

    return res


def show_colornames(cn_unit: int = 100):

    cn = np.zeros((cn_unit, 16*cn_unit, 3))

    for i in range(color_names.shape[0]):
        cn[:,i*cn_unit:(i+1)*cn_unit,:] = color_names[i]

    cv2.imshow("color names", cn)


def debackground(output_size: np.ndarray, image: np.ndarray) -> np.ndarray:

    '''
    In experimental, the get_grids() and bitwise() are the most 2 of time consuming method 
    
    To Do:
    Try to accelerate get_grids() & bitwise()

    Here is the bitwise() accelerate method:
    It's simple. I change the process flow.
    The reason that bitwise() cost time is because it would calculate every pixels. 
    Even the image is 640x480, it would takes 307200 calculation.
    I resize the original image to fit the mask shape and then do bitwise at the scale.
    Which is downsampling the image to decrease the calculation.
    '''

    # normalize the image
    image_norm = image / 255

    # t0 = time.time()
    # Get bins
    bins = get_grids(output_size=output_size, image=image_norm)
    # t1 = time.time()

    mask = connect_background(bins)
    # t2 = time.time()
    # mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    img_resize = cv2.resize(image_norm, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    # t3 = time.time()

    # Processin with mask
    # res = cv2.bitwise_and(image_norm, image_norm, mask=mask.astype('uint8'))
    # res = bitwise(image_norm, mask.astype('uint8'))
    res = bitwise(img_resize, mask.astype('uint8'))
    res = cv2.resize(res, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    # t4 = time.time()

    # print("get bins: {:.3f}ms, get mask: {:.3f}ms, resize mask: {:.3f}ms, get result: {:.3f}ms".format(
    #     (t1-t0)*1000, (t2-t1)*1000, (t3-t2)*1000, (t4-t3)*1000
    # ))

    return res


def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", 
                        help="Input source from WebCam, images or video. Default: Camera 0.")
    parser.add_argument("--show-colornames", action="store_true", 
                        help="Show all the setting color names.")
    parser.add_argument("--size", nargs="+", default=[40, 30],
                        help="Set mosaic/mask size. Default: [40, 30]")
    return parser.parse_args()


def main():

    args = parse_arguments()

    # Show color names
    if args.show_colornames:
        show_colornames()

    # if the source is WebCam
    # then convert the string to integer
    source = None
    if args.source.isdigit():
        source = int(args.source)

    else:
        # path to the video or parent directory of the images
        '''
        ToDo:
        List all the images to the source, making the VideoCapture can read the images from it 
        '''
        source = args.source 

    cap = None
    try:
        cap = cv2.VideoCapture(source)
    
    except Exception as e:
        get_error_info(e)
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    # 8, 6
    size = np.array(args.size)

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

        st_getgrids = time.time()

        # Get grids
        grids = get_grids(output_size=size, image=frame_norm)

        end_getgrids = time.time()
        exe_getgrids_time = end_getgrids-st_getgrids

        mask = connect_background(grids)
        # mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_AREA)
        img_resize = cv2.resize(frame_norm, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)

        end_getmask = time.time()
        exe_getmask_time = end_getmask - end_getgrids

        # Processin with mask
        # res1 = cv2.bitwise_and(frame_norm, frame_norm, mask=mask.astype('uint8'))
        res1 = cv2.bitwise_and(img_resize, img_resize, mask=mask.astype('uint8'))
        end_bitwise_cv = time.time()
        cv_bitwise_time = end_bitwise_cv - end_getmask

        res = bitwise(img_resize, mask.astype('uint8'))
        res = cv2.resize(res, (frame_norm.shape[1], frame_norm.shape[0]), interpolation=cv2.INTER_AREA)
        end_bitwise = time.time()

        custom_bitwise_time = end_bitwise - end_bitwise_cv

        if frame_counter > 1:
            # total_exe_time += exe_getgrids_time + exe_getmask_time + custom_bitwise_time
            total_exe_time += exe_getgrids_time + exe_getmask_time + cv_bitwise_time

        print('get grid time: %.3fms, get mask time: %.3fms, custom bitwise time: %.3fms, opencv bitwise time: %.3fms' % 
                (
                    exe_getgrids_time*1000, 
                    exe_getmask_time*1000,
                    custom_bitwise_time*1000,
                    cv_bitwise_time*1000
                )
             )

        # Show image
        concate = np.concatenate((frame_norm, res), axis=1)
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


if __name__ == '__main__':

    main()