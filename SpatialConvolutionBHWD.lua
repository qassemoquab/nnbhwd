local SpatialConvolution, parent = torch.class('nn.SpatialConvolutionBHWD', 'nn.Module')

local doc = 
[[This is the primary spatial convolution module.
It performs a 2D convolution with nOutputPlane 3D kernels of size (kW*kH*nInputPlane).

Usage : m = nn.SpatialConvolutionBHWD(nInputPlane, nOutputPlane, kW, kH, dW, dH, padleft, padright, padtop, padbottom)
- nInputPlane is the number of input planes.
- nOutputPlane is the number of output planes.
- kW/kH is the kernel width/height.
- dW/dH is the stride over the x/y dimension.
- padleft, padright, padtop, padbottom are the amount of zero-padding pixels.


It only works in BATCH MODE (4D) :
- with the following input layout : (batch, y, x, channels).
- channels are the contiguous dimension.
- a single image must be a (1, y, x, channels) tensor.

If self.noUnfold=true, then an alternate implementation will be used.


The module doesn't require fixed-size inputs but it will work faster in :
- trivial mode : kW=kH=dW=dH=1, paddings=0, (performs a per-pixel linear transform)
- fully-connected mode : kW = input width, kH = input height, paddings = 0, (performs global linear transform)
- otherwise it will use the standard convolution, that is based on BLAS GEMM.
The module switches between fully-connected and conv at training time if possible.

]]


function SpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padleft, padright, padtop, padbottom)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.padleft = padleft or 0
   self.padright = padright or 0
   self.padtop = padtop or 0
   self.padbottom = padbottom or 0
   self.tmpweight=torch.Tensor()
   self.tmpgradweight=torch.Tensor()

   self.weight = torch.Tensor(nOutputPlane, kH, kW, nInputPlane)
   self.bias = torch.Tensor(nOutputPlane)

   self.gradWeight = torch.Tensor(nOutputPlane, kH, kW, nInputPlane):zero()
   self.gradBias = torch.Tensor(nOutputPlane):zero()
   
   self:reset()
   
   self.mode='conv'
   self.gpucompatible = true
end





function SpatialConvolution:reset(stdv)
   --
   -- resets weights according to gaussian (0,stdv) distribution
   --
   if stdv then
      stdv = stdv
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   torch.randn(self.weight, self.weight:size())
   self.weight:mul(stdv)
   torch.randn(self.bias, self.bias:size())
   self.bias:mul(stdv)
end





function SpatialConvolution:optimize(input)
-- switches the module to optimize computation :
-- - case 1 : kernels are 1x1 : perform linear transform of features
-- - case 2 : kernels are same size as input : perform fully-connected linear transform
-- - case 3 : general case : perform convolution

   if self.padleft==0 and 
      self.padright==0 and 
      self.padtop==0 and 
      self.padbottom==0 and
      self.kH==1 and 
      self.kW==1 and 
      self.dH==1 and 
      self.dW==1 then 
      self.weight=self.weight:resize(self.nOutputPlane, self.nInputPlane)
      self.gradWeight=self.gradWeight:resize(self.nOutputPlane, self.nInputPlane)
      self.mode='trivial'
      return
   end
   if self.padleft==0 and 
      self.padright==0 and 
      self.padtop==0 and 
      self.padbottom==0 and 
      input:size(2)==self.kH and
      input:size(3)==self.kW and
      input:size(4)==self.nInputPlane then 
      self.mode='fc'
      return
   end 
   self.mode='conv'
   return
end




--
-- 3 functions to update outputs
--

function SpatialConvolution:updateOutputTrivial(input)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1)*input:size(2)*input:size(3), input:stride(3)})
   
   -- MM
   self.output:resize(input:size(1)*input:size(2)*input:size(3), self.weight:size(1))
   self.output:zero():addr(1, input.new(input:size(1)*input:size(2)*input:size(3)):fill(1), self.bias)
   self.output:addmm(1, tinput, self.weight:t())

   -- output is unflattened
   self.output:resize(input:size(1), input:size(2), input:size(3), self.weight:size(1))

end

function SpatialConvolution:updateOutputFC(input)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1), input:stride(1)})
   
   -- weight is flattened (view)
   local tweight=self.weight.new()
   tweight:set(self.weight:storage(), 1, torch.LongStorage{self.weight:size(1), self.weight:stride(1)})
   
   -- MM
   self.output:resize(input:size(1), self.weight:size(1))
   self.output:zero():addr(1, input.new(input:size(1)):fill(1), self.bias)
   self.output:addmm(1, tinput, tweight:t())
   self.output:resize(input:size(1), 1, 1, self.weight:size(1))
end

function SpatialConvolution:updateOutputConv(input)
	if self.noUnfold then 
	   input.nn.SpatialConvolutionBHWD_updateOutput(self, input)
	else
		input.nn.SpatialConvolutionUnfold_updateOutput(self, input)
	end
end


function SpatialConvolution:updateOutput(input)
   --
   -- find the best computation mode and run it to update outputs
   --
   if input:size(2)+self.padtop+self.padbottom<self.kH or input:size(3)+self.padleft+self.padright<self.kW then error ('input is too small') end
   self:optimize(input)
   if self.mode=='trivial' then self:updateOutputTrivial(input) end
   if self.mode=='fc' then self:updateOutputFC(input) end
   if self.mode=='conv' then self:updateOutputConv(input) end
   return self.output
end




--
-- 3 functions to update gradients
--


function SpatialConvolution:updateGradInputTrivial(input, gradOutput)
   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1)*gradOutput:size(2)*gradOutput:size(3), gradOutput:stride(3)})
  
   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)

   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end

   -- gradInput is flattened (view)
   local tgradInput=self.gradInput.new()
   tgradInput:set(self.gradInput:storage(), 1, torch.LongStorage{self.gradInput:size(1)*self.gradInput:size(2)*self.gradInput:size(3), self.gradInput:stride(3)})

   tgradInput:addmm(0, 1, tgradOutput, self.weight)
   
end

function SpatialConvolution:updateGradInputFC(input, gradOutput)
   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1), gradOutput:stride(1)})
   
   -- weight is flattened (view)
   local tweight=self.weight.new()
   tweight:set(self.weight:storage(), 1, torch.LongStorage{self.weight:size(1), self.weight:stride(1)})
   
   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end

   -- gradInput is flattened (view)
   local tgradInput=self.gradInput.new()
   tgradInput:set(self.gradInput:storage(), 1, torch.LongStorage{self.gradInput:size(1), self.gradInput:stride(1)})

   tgradInput:addmm(0, 1, tgradOutput, tweight)
end

function SpatialConvolution:updateGradInputConv(input, gradOutput)
	if self.noUnfold then 
   	input.nn.SpatialConvolutionBHWD_updateGradInput(self, input, gradOutput)
	else
		input.nn.SpatialConvolutionUnfold_updateGradInput(self, input, gradOutput)
	end
end

function SpatialConvolution:updateGradInput(input, gradOutput)
   --
   -- find the best computation mode and run it to update gradients
   --
   self:optimize(input)
   if self.mode=='trivial' then self:updateGradInputTrivial(input, gradOutput) end
   if self.mode=='fc' then self:updateGradInputFC(input, gradOutput) end
   if self.mode=='conv' then self:updateGradInputConv(input, gradOutput) end
   return self.gradInput

end




-- update weight gradients

function SpatialConvolution:accGradParametersTrivial(input, gradOutput, scale)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1)*input:size(2)*input:size(3), input:stride(3)})

   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1)*gradOutput:size(2)*gradOutput:size(3), gradOutput:stride(3)})
   
   self.gradWeight:addmm(scale, tgradOutput:t(), tinput)
   self.gradBias:addmv(scale, tgradOutput:t(), tinput.new(input:nElement()/self.weight:size(2)):fill(1))
end

function SpatialConvolution:accGradParametersFC(input, gradOutput, scale)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1), input:stride(1)})

   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1), gradOutput:stride(1)})
   
   -- weight is flattened (view)
   local tgradWeight=self.gradWeight.new()
   tgradWeight:set(self.gradWeight:storage(), 1, torch.LongStorage{self.gradWeight:size(1), self.gradWeight:stride(1)})

   tgradWeight:addmm(scale, tgradOutput:t(), tinput)
   self.gradBias:addmv(scale, tgradOutput:t(), tinput.new(input:size(1)):fill(1))
end

function SpatialConvolution:accGradParametersConv(input, gradOutput, scale)
	if self.noUnfold then 
   	input.nn.SpatialConvolutionBHWD_accGradParameters(self, input, gradOutput, scale) 
	else
		input.nn.SpatialConvolutionUnfold_accGradParameters(self, input, gradOutput, scale) 
	end
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
   --
   -- find the best computation mode and run it to update gradients, if the module is learning
   -- rescale by dividing by the number of examples in the batch
   --

   self:optimize(input)
   local scale = scale or 1
   if self.mode=='trivial' then self:accGradParametersTrivial(input, gradOutput, scale) end
   if self.mode=='fc' then self:accGradParametersFC(input, gradOutput, scale) end
   if self.mode=='conv' then self:accGradParametersConv(input, gradOutput, scale) end
end







local function tensorsizestring(t)
   local str = t:type() .. ' - '
   if t:dim()>0 then
      for i=1,t:dim()-1 do
         str = str .. t:size(i) .. 'x'
      end	
      str = str .. t:size(t:dim())
   else 
      str = str .. 'empty'
   end
   return str
end

function SpatialConvolution:__tostring__()
   local tab = '     |  '
   local line = '\n'
   local next = ' -> '
   local str = 'nn.SpatialConvolutionBHWD('
   str = str .. self.nInputPlane ..', '
   str = str .. self.nOutputPlane ..', '
   str = str .. self.kW ..', '
   str = str .. self.kH ..', '
   str = str .. self.dW ..', '
   str = str .. self.dH ..', '
   str = str .. self.padleft ..', '
   str = str .. self.padright ..', '
   str = str .. self.padtop ..', '
   str = str .. self.padbottom ..')'
   str = str .. line .. tab
	local learnstring
   if self.learn then 
		learnstring='yes' 
	else 
		learnstring='no'
	end
   str = str .. 'learning    : ' .. learnstring
   str = str .. line .. tab
   str = str .. 'output      : ' .. tensorsizestring(self.output)
   str = str .. line .. tab
   str = str .. 'gradInput   : ' .. tensorsizestring(self.gradInput)
   str = str .. line

   return str
end




