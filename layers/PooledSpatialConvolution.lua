require 'cudnn'
local PooledSpatialConvolution, parent = torch.class(
  'cudnn.PooledSpatialConvolution', 'nn.Module')

function PooledSpatialConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW,
                                         dH, padW, padH, poolSize, lambda,
                                         initialSparsity)
   assert(poolSize, "poolSize is required")

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.padW = padW
   self.padH = padH
   self.poolSize = poolSize
   self.lambda = lambda or opt.lambda or 0.1
   self.sparseTh = (initialSparsity or opt.initialSparsity or 0.01) / math.sqrt(
    self.kW * self.kH * self.poolSize)

   self.poolconv = cudnn.SpatialConvolution(nInputPlane, poolSize, 1, 1, 1, 1,
                                            padW, padH):noBias()
   self.alignconv = cudnn.SpatialConvolution(poolSize, nOutputPlane, kW, kH, dW,
                                             dH, 0, 0)
   self.m = nn.Sequential():add(self.poolconv):add(self.alignconv)

   self.weight = self.poolconv.weight
   self.bias = self.alignconv.bias
   self.gradWeight = self.poolconv.gradWeight
   self.gradBias = self.alignconv.gradBias

   self:reset()
end

function PooledSpatialConvolution:noBias()
   self.alignconv:noBias()
   self.bias = nil
   self.gradBias = nil
   return self
end

function PooledSpatialConvolution:reset()
   self.poolconv:reset()
   self.alignconv:reset()
   return self
end

-- Randomly re-initialize all non-zero elements of the align convolution
function PooledSpatialConvolution:resetAlignConv()
   self.alignconv:reset()
   if self.zeroMask then
      self.alignconv.weight[self.zeroMask] = 0
   end
end

function PooledSpatialConvolution:updateOutput(input)
   self.zeroMask = self.zeroMask or input.new():resizeAs(self.alignconv.weight)
   self.zeroMask:abs(self.alignconv.weight)
   self.zeroMask:le(self.zeroMask, self.sparseTh)
   self.alignconv.weight[self.zeroMask] = 0
   if opt.logSparsity then
      print (string.format("%s \t %0.4f sparse", self, self.zeroMask:sum() /
                           self.zeroMask:nElement()))
   end

   self.output = self.m:forward(input)
   return self.output
end

function PooledSpatialConvolution:updateGradInput(input, gradOutput)
   self.gradInput = self.m:updateGradInput(input, gradOutput)
   return self.gradInput
end

function PooledSpatialConvolution:accGradParameters(input, gradOutput, scale)
   self.m:accGradParameters(input, gradOutput, scale)
   if self.lambda ~= 0 then
      self.gradRegularizer = self.gradRegularizer or self.alignconv.weight.new(
        ):resizeAs(self.alignconv.weight)
      self.gradRegularizer:sign(self.alignconv.weight)

      local lambda = self.lambda * self.sparseTh
      self.alignconv.gradWeight:add(lambda, self.gradRegularizer)
   end

   -- backprop the gradient through the threshold function
   self.alignconv.gradWeight[self.zeroMask] = 0
end

function PooledSpatialConvolution:parameters()
   -- get the parameters of poolconv
   local params, grads = parent.parameters(self)
   -- add the align conv parameters
   table.insert(params, self.alignconv.weight)
   table.insert(grads, self.alignconv.gradWeight)

   return params, grads
end

function PooledSpatialConvolution:zeroGradParameters()
   self.gradWeight:fill(0)
   self.alignconv.gradWeight:fill(0)
   if self.gradBias then
      self.gradBias:fill(0)
   end
end

function PooledSpatialConvolution:clearState()
   self.m:clearState()
end

function PooledSpatialConvolution:__tostring__()
   return torch.type(self) ..
      string.format('(%d -> %d, %dx%d, pool%d)', self.nInputPlane,
                    self.nOutputPlane, self.kH, self.kW, self.poolSize)
end
