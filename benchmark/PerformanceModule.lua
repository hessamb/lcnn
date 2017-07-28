local Performance = torch.class('nn.Performance')

function Performance:__init(module)
   self.moduleType = torch.type(module)

   if torch.isTypeOf(module, nn.Container) then
      self:container(module)
   elseif torch.isTypeOf(module, nn.SpatialConvolution) then
      self:convolution(module)
   elseif torch.isTypeOf(module, nn.Linear) then
      self:linear(module)
   elseif torch.isTypeOf(module, nn.ReLU) or torch.isTypeOf(module, cudnn.ReLU) then
      self:relu(module)
   elseif torch.isTypeOf(module, cudnn.PooledSpatialConvolution) then
      self:sparseConvolution(module)
   elseif torch.isTypeOf(module, nn.SpatialAveragePooling) or torch.isTypeOf(module, cudnn.SpatialAveragePooling)
         or torch.isTypeOf(module, nn.SpatialMaxPooling) or torch.isTypeOf(module, cudnn.SpatialMaxPooling) then
      self:pooling(module)
   else
      self:freeLayer(module)
   end
end

function Performance:container(module)
   local containerFactory = torch.factory( torch.type(module) )
   local performanceFactory = torch.factory( torch.type(self) )

   self.submodule = containerFactory()
   self.submodule.modules = {}

   self.computation = 0
   self.memory = 0
   for i=1,module:size() do
      local performanceModule = performanceFactory()
      performanceModule:__init(module:get(i))
      self.submodule:add(performanceModule)
      self.computation = self.computation + self.submodule:get(i).computation
      self.memory = self.memory + self.submodule:get(i).memory
   end
end

function Performance:convolution(module)
   self.memory = module.weight:nElement() + (module.bias and module.bias:nElement() or 0)
   self.computation = module.output:nElement() * module.weight[1]:nElement()
end

function Performance:linear(module)
   self.memory = module.weight:nElement() + (module.bias and module.bias:nElement() or 0)
   self.computation = module.weight:nElement()
end

function Performance:relu(module)
   self:freeLayer(module)
end

function Performance:sparseConvolution(module)
   self.nsp = module.alignconv.weight:ne(0):sum() / module.alignconv.weight:nElement()
   self.memory = module.poolconv.weight:nElement() + module.alignconv.weight:nElement() * self.nsp
                  + (module.alignconv.bias and module.alignconv.bias:nElement() or 0)
   self.computation = module.poolconv.output:nElement() * module.poolconv.weight[1]:nElement()
                  + module.output:nElement() * module.alignconv.weight[1]:nElement() * self.nsp
   local poolcomp = module.poolconv.output:nElement() * module.poolconv.weight[1]:nElement()
   local aligncomp = module.output:nElement() * module.alignconv.weight[1]:nElement() * self.nsp
end

function Performance:pooling(module)
   self:freeLayer(module)
end

function Performance:freeLayer(module)
   self.memory = 0
   self.computation = 0
end

function Performance:__tostring__(totalmem, totalcomp)
   totalmem = totalmem or self.memory
   totalcomp = totalcomp or self.computation
   local str
   if self.submodule then
      local tab = '  '
      local line = '\n'
      local next = ' -> '
      str = self.moduleType
      str = str .. ' {' .. line .. tab .. '[input'
      for i=1,#self.submodule.modules do
         str = str .. next .. '(' .. i .. ')'
      end
      str = str .. next .. 'output]'
      for i=1,#self.submodule.modules do
         str = str .. line .. tab .. '(' .. i .. '): '
               .. self.submodule.modules[i]:__tostring__(totalmem,totalcomp)
                                           :gsub(line, line .. tab)
      end
      str = str .. line .. '}'
   else
      str = self.moduleType
   end
   return str .. string.format(' MEM=%d(%.2f%%) COMP=%d(%.2f%%)', self.memory,
   100*self.memory/totalmem, self.computation, 100*self.computation/totalcomp)
end
