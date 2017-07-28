--  Modified by Hessam Bagherinezhad (XNOR AI)
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 Imagenet Training script')
   cmd:text()
   cmd:text('Options:')
   ------------ General options --------------------

   cmd:option('-cache', './cache/', 'subdirectory in which to save/log experiments')
   cmd:option('-data', 'data/', 'Home of the dataset')
   cmd:option('-dataset',  'imagenet', 'Dataset Name: imagenet is the only option')
   cmd:option('-manualSeed',         2, 'Manually set RNG seed')
   cmd:option('-GPU',                1, 'Default preferred GPU')
   cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
   cmd:option('-noCheckpoint',   false, 'Don\'t save model checkpoints')
   cmd:option('-testOnly',    false, 'only perform one validation epoch and exit')
   cmd:option('-benchmark', false ,  'benchmark the loaded model and exit')
   cmd:option('-logSparsity', false ,  'print out the sparity on each layer of the network')
   cmd:option('-nLastIterAccuracy', 100 ,  'number of last iterations for reporting the "recent accuracy"')
   ------------- Data options ------------------------
   cmd:option('-nDonkeys',        8, 'number of donkeys to use (data loading threads)')
   cmd:option('-imageSize',         256,    'Smallest side of the resized image')
   cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
   cmd:option('-nClasses',        1000, 'number of classes in the dataset')
   ------------- Training options --------------------
   cmd:option('-nEpochs',         55,    'Number of total epochs to run')
   cmd:option('-epochSize',       10000, 'Number of batches per epoch')
   cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       128,   'mini-batch size (1 = pure stochastic)')
   cmd:option('-breakBatch',      1,     'break each batch to this many pieces to fit into memory')
   ---------- Optimization options ----------------------
   cmd:option('-LR',    1.0, 'learning rate multiplier; multiplies the default LR regime with this number')
   cmd:option('-momentum',        0.9,  'optimizer momentum')
   cmd:option('-weightDecay',     0, 'weight decay')
   cmd:option('-shareGradInput',  true, 'Sharing the gradient memory for resnet')
   cmd:option('-updateLRregime',  false, 'Update lr regime of the model loaded with the one in the model definition')
   cmd:option('-constantLR', -1 ,  'keep learning constantly this value')
   ---------- Model options ----------------------------------
   cmd:option('-netType',     'alexnet-lcnn', 'Options: alexnet | alexnet-lcnn | resnet | resnet-lcnn')
   cmd:option('-criterion',     'classnll', 'Options: clasnll | ranking')
   cmd:option('-depth',        18,       'ResNet depth: 18 | 34 | 50 | 101 | ...', 'number')
   cmd:option('-shortcutType', '',       'Options: A | B | C')
   cmd:option('-optimType',       'sgd', 'Options: sgd | adam')
   cmd:option('-retrain',     'none', 'provide path to model to retrain with')
   cmd:option('-resume',      'none', 'resume training for a directory')
   cmd:option('-loadParams',  'none', 'provide path to model to load the parameters')
   cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
   cmd:option('-shortcutType',  'B', 'type of short cut in resnet: A|B|C')
   cmd:option('-dropout', 0.5 , 'Dropout ratio')
   cmd:option('-poolSize',   100 , 'poolSize of PooledSpatialConvolution')
   cmd:option('-pool1',   100 , 'poolSize of 1st block types of resnet')
   cmd:option('-pool2',   100 , 'poolSize of 2nd block types of resnet')
   cmd:option('-pool3',   100 , 'poolSize of 3rd block types of resnet')
   cmd:option('-pool4',   100 , 'poolSize of 4th block types of resnet')
   cmd:option('-lambda',       0 , 'The regularization factor for PooledSpatialConvolution')
   cmd:option('-firstPool', 3 ,     'The size of vector pool for the first layer')
   cmd:option('-fcPool',  1024 ,   'The size of dictionary for the fully connected layers')
   cmd:option('-classifierPool', 0 ,  'Pool size for the classification layer')
   cmd:option('-initialSparsity', 0.01 ,  'initial sparsity of pooled convolutional layers')

   cmd:text()

   local opt = cmd:parse(arg or {})
   -- add commandline specified options
   opt.save = paths.concat(opt.cache,
                  cmd:string(opt.netType, opt,
                  {netType=true, retrain=true, loadParams=true, optimState=true,
                  cache=true, data=true, nEpochs=true, noCheckpoint=true}))

   -- add date/time
   opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))

   if opt.breakBatch > 1 then
      assert(opt.batchSize % opt.breakBatch == 0, "batchSize must be divisible by breakBatch")
      opt.batchSize = opt.batchSize / opt.breakBatch
      opt.epochSize = opt.epochSize * opt.breakBatch
      opt.nLastIterAccuracy = opt.nLastIterAccuracy * opt.breakBatch
   end
   
   local logfile = paths.concat(opt.save, 'log.txt')
   if opt.resume ~= 'none' then
      opt.save = opt.resume
      logfile = io.open(paths.concat(opt.save, 'log.txt'), 'a')
      opt.epochNumber = 1
      for i, fname in ipairs(paths.dir(opt.save)) do
         local enumber = fname:match('optimState_(%d*)%.t7')
         if enumber and tonumber(enumber) + 1 > opt.epochNumber then
            opt.epochNumber = tonumber(enumber) + 1
         end
      end
      if opt.epochNumber > 1 then -- a saved model found in opt.resume
         opt.retrain = string.format('%s/model_%d.t7', opt.save, opt.epochNumber-1)
         opt.optimState = string.format('%s/optimState_%d.t7', opt.save, opt.epochNumber-1)
      end
   end
   paths.mkdir(opt.save)
   cmd:log(logfile, opt)
   return opt
end

return M
