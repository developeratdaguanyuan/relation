require 'nn'
require 'rnn'
require 'DataLoader'
require 'RelationLSTM'


local cmd = torch.CmdLine()
cmd:option('-trainDataFile', '../data/question_relationID_train_tiny.txt', 'training data file')
cmd:option('-validDataFile', '../data/question_relationID_valid_tiny.txt', 'validation data file')
cmd:option('-wordEmbeddingFile', '../data/embedding.txt')
cmd:option('-modelDirectory', "../model")
cmd:option('-useEmbed', 1, 'whether to use word embedding')
cmd:option('-useGPU', 1, 'whether to use GPU')
cmd:option('-learningRate', 0.01, 'learning rate')
cmd:option('-costMargin', 1, 'margin of hinge loss')
cmd:option('-vocabularySize', 313939, 'vocabulary size')
cmd:option('-vocabularyDim', 300, 'word embedding size')
cmd:option('-relationSize', 1837, 'relation size')
cmd:option('-relationDim', 256, 'relation embedding size')
cmd:option('-batchSize', 10, 'number of data in a batch')
cmd:option('-printEpoch', 10, 'print training loss every printEpoch iterations')

local opt = cmd:parse(arg)
local relationLSTM = RelationLSTM(opt)
relationLSTM:train()

