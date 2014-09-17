local SpatialMaxPoolingBHWD, parent = torch.class('nn.SpatialMaxPoolingBHWD', 'nn.Module')

local help_str = 
[[This is the max-pooling module.
It performs a 2D max-pooling within cells of size (poolW,poolH), and stride (dW,dH).

Usage : m = nxn.SpatialMaxPooling(poolW, poolH, dW, dH)

It only works in BATCH MODE (4D) :
- with the following input layout : (batch, y, x, channels).
- channels are the contiguous dimension.
- a single image must be a (1, y, x, channels) tensor.

The module doesn't require fixed-size inputs.]]


function SpatialMaxPoolingBHWD:__init(poolW, poolH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.poolW = poolW
   self.poolH = poolH
   self.dW = dW
   self.dH = dH
   self.indices = torch.Tensor()
   self.gpucompatible = true
end


function SpatialMaxPoolingBHWD:updateOutput(input)
   input.nn.SpatialMaxPoolingBHWD_updateOutput(self, input)
   return self.output
end

function SpatialMaxPoolingBHWD:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.SpatialMaxPoolingBHWD_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

