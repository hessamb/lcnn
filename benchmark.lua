function benchmark()
   requireAll 'benchmark'

   model:evaluate()

   local single_model = cleanDPT(model)
   local input = trainLoader:sample(1):cuda()

   single_model:forward(input)
   performance = nn.Performance(single_model)

   print (performance)
   print (string.format('Benchmark Summary: memory %d computation %d',
                           performance.memory, performance.computation))
end
