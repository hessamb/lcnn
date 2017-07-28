function createModel()
   require 'cudnn'

   local ConvFull = cudnn.SpatialConvolution
   local ConvPool = cudnn.PooledSpatialConvolution
   local SBN = nn.SpatialBatchNormalization
   local Dropout = nn.Dropout
   local ReLU = cudnn.ReLU
   local Max = nn.SpatialMaxPooling

   local features = nn.Sequential()
   features:add(ConvPool(3,64,11,11,4,4,2,2, opt.firstPool))       -- 224 -> 55
   features:add(SBN(64,1e-3))
   features:add(ReLU(true))
   features:add(Max(3,3,2,2))                   -- 55 ->  27
   features:add(ConvPool(64,192,5,5,1,1,2,2, opt.poolSize))       --  27 -> 27
   features:add(SBN(192,1e-3))
   features:add(ReLU(true))
   features:add(Max(3,3,2,2))                   --  27 ->  13
   features:add(ConvPool(192,384,3,3,1,1,1,1, opt.poolSize))      --  13 ->  13
   features:add(SBN(384,1e-3))
   features:add(ReLU(true))
   features:add(ConvPool(384,256,3,3,1,1,1,1, opt.poolSize))      --  13 ->  13
   features:add(SBN(256,1e-3))
   features:add(ReLU(true))
   features:add(ConvPool(256,256,3,3,1,1,1,1, opt.poolSize))      --  13 ->  13
   features:add(SBN(256,1e-3))
   features:add(ReLU(true))
   features:add(Max(3,3,2,2))                   -- 13 -> 6

   local classifier = nn.Sequential()
   classifier:add(Dropout(opt.dropout))
   if opt.fcPool > 0 then
      -- Implement fully connected layers via a 1x1 convolution
      classifier:add(ConvPool(256,4096,6,6,1,1,0,0, opt.fcPool))
   else
      classifier:add(ConvFull(256,4096,6,6))
   end
   classifier:add(SBN(4096, 1e-3))
   classifier:add(ReLU())

   classifier:add(Dropout(opt.dropout))
   if opt.fcPool > 0 then
      classifier:add(ConvPool(4096,4096,1,1,1,1,0,0, opt.fcPool))
   else
      classifier:add(ConvFull(4096,4096,1,1))
   end
   classifier:add(SBN(4096, 1e-3))
   classifier:add(ReLU())

   if opt.classifierPool > 0 then
      classifier:add(ConvPool(4096,nClasses,1,1,1,1,0,0, opt.classifierPool))
   else
      classifier:add(ConvFull(4096,nClasses,1,1))
   end
   classifier:add(nn.View(nClasses))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential():add(features):add(classifier)

   return model
end
