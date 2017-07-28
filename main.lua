debugger = require 'fb.debugger'
require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'gnuplot'

torch.setdefaulttensortype('torch.FloatTensor')

local opts = paths.dofile('opts.lua')

opt = opts.parse(arg)

nClasses = opt.nClasses

paths.dofile('util.lua')

requireAll 'layers'

print(opt)

cutorch.setDevice(opt.GPU) -- by default, use GPU 1
torch.manualSeed(opt.manualSeed)

print('Saving everything to: ' .. opt.save)

paths.dofile('data.lua')
paths.dofile('model.lua')
paths.dofile('train.lua')
paths.dofile('test.lua')

paths.dofile('benchmark.lua')
if opt.benchmark then
	benchmark()
else
	epoch = opt.epochNumber
	if opt.testOnly then
		test()
	else
		for i=opt.epochNumber,opt.nEpochs do
			train()
			test()
			epoch = epoch + 1
		end
	end
end
