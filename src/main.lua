require 'DataLoader'

local dataLoader = DataLoader("../data/question_relationID_train.txt", 10)
local data, mark = unpack(dataLoader:nextBatch())
print(data)
