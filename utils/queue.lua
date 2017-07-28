local Queue = torch.class('Queue')

function Queue:__init()
   self._start = 1
   self._end = 1
   self.list = {}
end

function Queue:pushleft(item)
   self.list[self._start - 1] = item
   self._start = self._start - 1
end

function Queue:pushright(item)
   self.list[self._end] = item
   self._end = self._end + 1
end

function Queue:size()
   return self._end - self._start
end

function Queue:empty()
   return self:size() <= 0
end

function Queue:left()
   return self:empty() and nil or self.list[self._start]
end

function Queue:right()
   return self:empty() and nil or self.list[self._end - 1]
end

function Queue:popleft()
   if self:empty() then
      return nil
   else
      local res = self:left()
      self.list[self._start] = nil
      self._start = self._start + 1
      return res
   end
end

function Queue:popright()
   if self:empty() then
      return nil
   else
      local res = self:right()
      self.list[self._end - 1] = nil
      self._end = self._end - 1
      return res
   end
end

function Queue:iterate()
   local i = self._start
   return function()
      if i >= self._end then
         return nil
      else
         i = i + 1
         return self.list[i - 1]
      end
   end
end

function Queue:sum()
   local res = 0
   for i in self:iterate() do
      res = res + i
   end
   return res
end
return Queue
