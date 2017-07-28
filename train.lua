--  Modified by Mohammad Rastegari (Allen Institute for Artificial Intelligence (AI2))
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
require 'utils.queue'
--[[
1. Setup SGD optimization state and learning rate schedule
2. Create loggers.
3. train - this function handles the high-level training loop,
i.e. load data, train model, save model and state to disk
4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch, optimType)
   local regimes = {
      -- start, end,    LR,   WD,
      {  1,     18,   1e-2,   0 },
      { 19,     29,   5e-3,   0  },
      { 30,     43,   1e-3,   0 },
      { 44,     52,   5e-4,   0 },
      { 53,    1e8,   1e-4,   0 },
   }

   local params, newRegime
   for _, row in ipairs(regimes) do
      if epoch >= row[1] and epoch <= row[2] then
         params, newRegime = { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
         break
      end
   end
   params.learningRate = params.learningRate * opt.LR
   return params, newRegime
end

paramsForEpoch = model.LRregime or paramsForEpoch

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
   learningRate = opt.LR,
   learningRateDecay = 0.0,
   momentum = opt.momentum,
   dampening = 0.0,
   weightDecay = opt.weightDecay,
}

if opt.optimState ~= 'none' then
   assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
   print('Loading optimState from file: ' .. opt.optimState)
   optimState = torch.load(opt.optimState)
   collectgarbage()
end

-- 2. Create loggers.
trainLogger = optim.Logger() -- make a logger with no file descriptor and then set it to a concat file descriptor
trainLogger.file = io.open(paths.concat(opt.save, 'train.log'), 'a')
local batchNumber
local top1Sum, top5Sum, loss_epoch
local top1Queue, top5Queue = Queue(), Queue() -- for keeping the last k iteration accuracies

if not trainLoader then
   require 'datasets.donkey'
end
-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch, opt.optimType)
   if newRegime or optimState.learningRate ~= params.learningRate then
      optimState.learningRate = params.learningRate
      optimState.learningRateDecay = 0.0
      optimState.momentum = opt.momentum
      optimState.dampening = 0.0
      optimState.weightDecay = params.weightDecay
   end

   if opt.constantLR >= 0 then
      optimState.learningRate = opt.constantLR
   end


   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1Sum = 0
   top5Sum = 0
   loss_epoch = 0

   model:zeroGradParameters()
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function()
         local inputs, labels = trainLoader:sample(opt.batchSize)
         return inputs, labels
      end,
      -- the end callback (runs in the main thread)
      trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()


   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1Sum/opt.epochSize,
      ['% top5 accuracy (train set)'] = top5Sum/opt.epochSize,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
   .. 'average loss (per batch): %.2f \t '
   .. 'accuracy(%%):\t top-1 %.2f\t',
   epoch, tm:time().real, loss_epoch, top1Sum/opt.epochSize))
   print('\n')

   -- save model
   collectgarbage()

   if not opt.noCheckpoint then
      saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
      torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
   end
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
collectgarbage()
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()
local procTimer = torch.Timer()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   procTimer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs

   outputs = model:forward(inputs)
   err = criterion:forward(outputs, labels)

   local pred = outputs:float()

   local gradOutputs = criterion:backward(outputs, labels)
   model:backward(inputs, gradOutputs)

   if (batchNumber+1) % opt.breakBatch == 0 then -- should update the parameters and reset the gradients
      local feval = function()
         return err, gradParameters
      end

      if opt.breakBatch > 1 then -- Average gradients over sub-batches
         gradParameters:div(opt.breakBatch)
      end

      local optimizer = optim[opt.optimType]
      optimizer(feval, parameters, optimState)
      model:zeroGradParameters()
   end
   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end


   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err

   local top1, top5 = computeScore(pred, labels, 1)
   top1Sum = top1Sum + top1
   top5Sum = top5Sum + top5
   top1Queue:pushright(top1)
   top5Queue:pushright(top5)
   while top1Queue:size() > opt.nLastIterAccuracy do
      top1Queue:popleft()
   end
   while top5Queue:size() > opt.nLastIterAccuracy do
      top5Queue:popleft()
   end
   local top1RecentMean = top1Queue:sum() / top1Queue:size()
   local top5RecentMean = top5Queue:sum() / top5Queue:size()
   -- Calculate top-1 error, and print information
   print(('Epoch: [%d][%d/%d]\tTime %.3f(%.3f) Err %.4f Top1-%%: %.2f {%.2f, %.2f}'
            .. ' Top5-%%: %.2f {%.2f, %.2f} LR %.0e DataTime %.3f'):format(
               epoch, math.ceil(batchNumber / opt.breakBatch), opt.epochSize / opt.breakBatch,
               timer:time().real ,procTimer:time().real ,err, top1, top1Sum/batchNumber,
               top1RecentMean, top5, top5Sum/batchNumber, top5RecentMean,
               optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
   timer:reset()
end
