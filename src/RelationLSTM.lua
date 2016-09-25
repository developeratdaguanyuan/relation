require 'rnn'
require 'DataLoader'
require 'utils'

local RelationLSTM = torch.class('RelationLSTM')

function RelationLSTM:__init(opt)
  if opt.useGPU > 0 then
    require 'cunn'
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end

  self.batchSize = opt.batchSize
  self.trainDataPath = opt.trainDataFile
  self.dataLoader = DataLoader(self.trainDataPath, self.batchSize)

  self.vocabularySize = opt.vocabularySize
  self.vocabularyDim = opt.vocabularyDim
  self.relationSize = opt.relationSize
  self.relationDim = opt.relationDim
  self.hiddenSize = self.vocabularyDim
  self.costMargin = opt.costMargin

  self.linkEmbedding = nn.LookupTable(self.relationSize, self.relationDim)
  self.wordEmbedding = nn.LookupTable(self.vocabularySize, self.vocabularyDim)
  self.rnn = nn.Sequential()
            :add(nn.LSTM(self.hiddenSize, self.hiddenSize))
  self.encoder = nn.Sequential()
            :add(self.wordEmbedding)
            :add(nn.SplitTable(1, 2))
            :add(nn.Sequencer(self.rnn))
            :add(nn.SelectTable(-1))
            :add(nn.Linear(self.hiddenSize, self.relationDim))

  self.encoderModel = cudacheck(self.encoder)
  self.positiveEncoder, self.negativeEncoder
    = unpack(cloneModulesWithSharedParameters(self.encoderModel, 2))

  self.linkEmbeddingModel = cudacheck(self.linkEmbedding)
  self.positiveLink, self.negativeLink
    = unpack(cloneModulesWithSharedParameters(self.linkEmbeddingModel, 2))
  self.positiveDotProduct = cudacheck(nn.DotProduct())
  self.negativeDotProduct = cudacheck(nn.DotProduct())

  self.criterion = cudacheck(nn.MarginRankingCriterion(self.costMargin))
end

function RelationLSTM:train()
  local accum_loss = 0
  for i = 1, 10000000 do
    self.encoderModel:zeroGradParameters()
    self.linkEmbeddingModel:zeroGradParameters()

    local data, pos_label, neg_label = unpack(self.dataLoader:nextBatch())

    local pos_data = self.positiveEncoder:forward(data)
    local neg_data = self.negativeEncoder:forward(data)
--    print(pos_data)
--    print(neg_data)

    local pos_link = self.positiveLink:forward(pos_label)
    local neg_link = self.negativeLink:forward(neg_label)
--    print(pos_link)
--    print(neg_link)

    local pos_score = self.positiveDotProduct:forward({pos_data, pos_link})
    local neg_score = self.negativeDotProduct:forward({neg_data, neg_link})
--    print(pos_score)
--    print(neg_score)

    local loss =
      self.criterion:forward({pos_score, neg_score}, torch.Tensor(self.batchSize):fill(1))

--    print(loss)
    accum_loss = accum_loss + loss
    if (i % 100 == 0) then
      print(accum_loss)
      accum_loss = 0
    end

    local dev_score =
      self.criterion:backward({pos_score, neg_score}, torch.Tensor(self.batchSize):fill(1))

    local dev_pos_data, dev_pos_link =
      unpack(self.positiveDotProduct:backward({pos_data, pos_link}, dev_score[1]))
    local dev_neg_data, dev_neg_link =
      unpack(self.negativeDotProduct:backward({neg_data, neg_link}, dev_score[2]))

    self.positiveLink:backward(pos_label, dev_pos_link)
    self.positiveEncoder:backward(data, dev_pos_data)
    self.negativeLink:backward(neg_label, dev_neg_link)
    self.negativeEncoder:backward(data, dev_neg_data)

    self.encoderModel:updateParameters(0.01)
    self.linkEmbeddingModel:updateParameters(0.01)
  end
end

