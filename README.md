nnbhwd
======

These are two BHWD-contiguous modules for Torch7.
You have to install them on top of nn, cutorch and cunn.

The convolution :
- does an unfold/SGEMM 
- is very fast in some cases (when the number of filters becomes high).
- is very memory-hungry, and will work faster if there is a lot of free memory.
- should not crash if the amount of free memory is low (it tries to adjust itself)
- survives all kinds of paddings, strides, kernel sizes...


How to install :
- clone the repo
- do "luarocks make"
- require 'nnbhwd' in torch
- use
``` 
nn.SpatialConvolutionBHWD(nInputPlane, nOutputPlane, kW, kH, dW, dH, padleft, padright, padtop, padbottom)
nn.SpatialMaxPoolingBHWD(poolW, poolH, dW, dH)
```


These modules will become obsolete very soon, when nvidia finishes implementing cuDNN.

If you use these, please cite 

```
    @TechReport{Oquab13,
    author = "Oquab, M. and Bottou, L. and Laptev, I. and Sivic, J.",
    title = "Weakly Supervised Object Recognition with Convolutional Neural Networks",
    institution  = "INRIA",
    year = "2014",
    number = "HAL-01015140"
    }
```
