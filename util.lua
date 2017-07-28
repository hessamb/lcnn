--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2))
require 'optim'
local ffi=require 'ffi'

function computeScore(output, target, nCrops, confMatrix)
   if nCrops > 1 then
      -- Sum over crops
      output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
      --:exp()
      :sum(2):squeeze(2)
   end

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending
   if confMatrix then
      for i=1,output:size(1) do
         confMatrix[predictions[i][1]][target[i]] = confMatrix[predictions[i][1]][target[i]] + 1
      end
   end
   -- Find which predictions match the target
   local correct = predictions:eq(
   target:long():view(batchSize, 1):expandAs(output))

   local top1 = correct:narrow(2, 1, 1):sum() / batchSize
   local top5 = correct:narrow(2, 1, 5):sum() / batchSize

   return top1 * 100, top5 * 100
end

function getPR(confMatrix, s, e)
   if not confMatrix then
      return 0, 0
   end
   local correct = confMatrix[{{s,e}, {s,e}}]:trace()
   local nSample = confMatrix[{{}, {s,e}}]:sum()
   local nPredict = confMatrix[{{s,e}, {}}]:sum()

   local precision = nSample > 0 and correct / nSample or 0
   local recall = nPredict > 0 and correct / nPredict or 0
   print (correct, nSample, nPredict)
   return precision * 100, recall * 100
end

function makeDataParallel(model, nGPU)

   if nGPU > 1 then
      print('converting module to nn.DataParallelTable')
      assert(nGPU <= cutorch.getDeviceCount(), 'number of GPUs less than nGPU specified')
      local model_single = model
      model = nn.DataParallelTable(1)
      for i=1, nGPU do
         cutorch.setDevice(i)
         model:add(model_single:clone():cuda(), i)
      end
      model.LRregime = model_single.LRregime
   end
   cutorch.setDevice(opt.GPU)

   return model
end

function cleanDPT(module)
   if torch.type(model) == 'nn.DataParallelTable' then
      return module:get(1)
   else
      return module
   end
end

function saveDataParallel(filename, model)
   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(filename, cleanDPT(model):clearState())
   elseif torch.type(model) == 'nn.Sequential' then
      torch.save(filename, model:clearState())
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

function loadParams(model,saved_model)
   local params = model:parameters()
   local saved_params = saved_model:parameters()
   if params then
      for i=1,#params do
         params[i]:copy(saved_params[i])
      end
   end
   local bn= model:findModules("nn.SpatialBatchNormalization")
   local saved_bn= saved_model:findModules("nn.SpatialBatchNormalization")
   for i=1,#bn do
      bn[i].running_mean:copy(saved_bn[i].running_mean)
      bn[i].running_var:copy(saved_bn[i].running_var)
   end
end


function updateBinaryGradWeight(convNodes)
   if not opt.noScaleWeights then
      local start = opt.binaryFirst and 1 or 2
      local finish = opt.binaryLast and #convNodes or #convNodes-1
      for i = start, finish do
         local n = convNodes[i].weight[1]:nElement()
         local s = convNodes[i].weight:size()
         local m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n):expand(s);
         m[convNodes[i].weight:le(-1)]=0;
         m[convNodes[i].weight:ge(1)]=0;
         m:add(1/(n)):mul(1-1/s[2])
         convNodes[i].gradWeight:cmul(m)
      end
   end
   if opt.nGPU >1 then
      model:syncParameters()
   end
end

function printGradStatistics(convNodes)
   if not sum then
      sum = {}
   end
   for i = 1, #convNodes do
      if not sum[i] then
         sum[i] = 0
      end
      local gw = convNodes[i].gradWeight
      local gb = convNodes[i].gradInput
      sum[i] = sum[i] + gw:norm(1)
      print (string.format('%s -- [%0.6f - %0.6f] ~ %0.6f \t [%0.6f - %0.6f] ~ %0.6f',
                           convNodes[i], gw:min(), gw:max(), torch.abs(gw):mean(),
                           gb:min(), gb:max(), torch.abs(gb):mean()))
      print (sum[i])
   end
end

function printStatistics(params)
   if not sum then
      sum = {}
   end
   for i = 1, #params do
      if not sum[i] then
         sum[i] = 0
      end
      local p = params[i]
      local meanAbs = p:norm(1) / p:nElement()
      sum[i] = sum[i] + meanAbs
      local inspect = require 'inspect'
      print (string.format('%s -- [%0.6f - %0.6f] ~ %0.6f == %0.6f',
                           inspect(p:size():totable()), p:min(), p:max(), meanAbs, sum[i]))
   end
end


function meancenterConvParms(convNodes)
   local start = opt.binaryFirst and 1 or 2
   local finish = opt.binaryLast and #convNodes or #convNodes-1
   for i = start, finish do
      local s = convNodes[i].weight:size()
      local negMean = convNodes[i].weight:mean(2):mul(-1):repeatTensor(1,s[2],1,1);
      convNodes[i].weight:add(negMean)
   end
   if opt.nGPU >1 then
      model:syncParameters()
   end
end


function binarizeConvParms(convNodes)
   local start = opt.binaryFirst and 1 or 2
   local finish = opt.binaryLast and #convNodes or #convNodes-1
   for i = start, finish do
      local m
      if not opt.noScaleWeights then
         local n = convNodes[i].weight[1]:nElement()
         local s = convNodes[i].weight:size()
         m = convNodes[i].weight:norm(1,4):sum(3):sum(2):div(n):expand(s)
      end
      convNodes[i].weight:sign()
      if not opt.noScaleWeights then
         convNodes[i].weight:cmul(m)
      end

   end
   if opt.nGPU >1 then
      model:syncParameters()
   end
end


function clampConvParms(convNodes)
   local start = opt.binaryFirst and 1 or 2
   local finish = opt.binaryLast and #convNodes or #convNodes-1
   for i = start, finish do
      convNodes[i].weight:clamp(-1,1)
   end
   if opt.nGPU >1 then
      model:syncParameters()
   end
end

function rand_initialize(layer)
   local tn = torch.type(layer)
   if tn == "cudnn.SpatialConvolution" then
      local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
      layer.weight:copy(torch.randn(layer.weight:size()) * c)
      if layer.bias then layer.bias:fill(0) end
   elseif tn == "nn.SpatialConvolution" then
      local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
      layer.weight:copy(torch.randn(layer.weight:size()) * c)
      if layer.bias then layer.bias:fill(0) end
   elseif tn == "nn.BinarySpatialConvolution" then
      local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
      layer.weight:copy(torch.randn(layer.weight:size()) * c)
      if layer.bias then layer.bias:fill(0) end
   elseif tn == "nn.SpatialConvolutionMM" then
      local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
      layer.weight:copy(torch.randn(layer.weight:size()) * c)
      if layer.bias then layer.bias:fill(0) end
   elseif tn == "cudnn.VolumetricConvolution" then
      local c  = math.sqrt(2.0 / (layer.kH * layer.kW * layer.nInputPlane));
      layer.weight:copy(torch.randn(layer.weight:size()) * c)
      if layer.bias then layer.bias:fill(0) end
   elseif tn == "nn.Linear" then
      local c =  math.sqrt(2.0 / layer.weight:size(2));
      layer.weight:copy(torch.randn(layer.weight:size()) * c)
      if layer.bias then layer.bias:fill(0) end
   elseif tn == "nn.SpatialBachNormalization" then
      layer.weight:fill(1)
      if layer.bias then layer.bias:fill(0) end
   elseif tn == "cudnn.SpatialBachNormalization" then
      layer.weight:fill(1)
      if layer.bias then layer.bias:fill(0) end
   end
   -- if layer.weight then
   --    layer.weight:mul(10)
   -- end
end

function getBinaryMask(convNodes, params)
   local mask = params.new():resizeAs(params):fill(0)
   local start = opt.binaryFirst and 1 or 2
   local finish = opt.binaryLast and #convNodes or #convNodes-1
   for i = start, finish do
      local idx = convNodes[i].weight:data() - params:data()
      local len = convNodes[i].weight:nElement()
      mask[{{idx+1, idx+len}}]:fill(1)
   end
   return mask
end


function plot(x, nPoints)
   assert(x:nDimension() == 1)
   nPoints = nPoints or math.huge
   if nPoints > x:size(1) then
      nPoints = x:size(1)
   end

   local xx = x.new():resize(nPoints)
   local step = math.floor((x:size(1) - 1) / (nPoints - 1))
   for i=1,x:size(1),step do
      xx[math.ceil(i / step)] = x[i]
   end
   gnuplot.plot(xx)
end

function endsWith(str, suffix)
   return str:sub(str:len() - suffix:len() + 1) == suffix
end

function requireAll(dir)
   for i, fname in ipairs(paths.dir(dir)) do
      if endsWith(fname, '.lua') then
         require (dir .. '.' .. paths.basename(fname, '.lua'))
      end
   end
end

function sample(n, k)
   return torch.randperm(n)[{{1, k}}]
end

function getValues(tab)
   local values = {}
   for i,val in pairs(tab) do
      table.insert(values, val)
   end
   return values
end

function table2tensor(tab)
   local res = torch.Tensor(#tab)
   for i,v in ipairs(tab) do
      res[i] = v
   end
   return res
end
