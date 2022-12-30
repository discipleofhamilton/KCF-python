# To Do List

1. write a doc to introduce the method of `debackground`, what's it for and how does it work?
2. introduce the accelerate library `numba`, how does it work and how to use it?
3. next improvement:
   1. Keep improving the long-term tracking robustness
   2. Accelerate the debackground method

## Long-Term Tracking Robustness


## Accelerate Debackground Method

Here the 4 steps of the debackground:

1. Splite image into grids
2. Connect the background into one object which means to get the mask
3. Resize the mask
4. Filter the background through the mask

In experimental, the step 1 and 4 are the most time consuming:
* Splite image into grids takes about mean 10ms
* Filter the background through the mask takes about mean 8ms