require 'rnn'
require 'DataLoader'
require 'utils'
require 'logroll'

local RelationLSTM = torch.class('RelationLSTM')

function RelationLSTM:__init(opt)
  self.log = logroll.file_logger('logs/info.log')

  if opt.useGPU == 1 then
    require 'cunn'
    require 'cutorch'
    torch.setdefaulttensortype('torch.CudaTensor')
  else
    torch.setdefaulttensortype('torch.FloatTensor')
  end

  self.trainDataPath = opt.trainDataFile
  self.validDataPath = opt.validDataFile

  self.learningRate = opt.learningRate
  self.costMargin = opt.costMargin
  self.vocabularySize = opt.vocabularySize
  self.vocabularyDim = opt.vocabularyDim
  self.relationSize = opt.relationSize
  self.relationDim = opt.relationDim
  self.hiddenSize = self.vocabularyDim
  self.batchSize = opt.batchSize

  self.maxEpochs = opt.maxEpochs
  self.printEpoch = opt.printEpoch

  self.dataLoader
    = DataLoader(self.trainDataPath, self.batchSize, self.vocabularySize, self.relationSize)

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
  local epochLoss, accumLoss = 0, 0
  local maxIter = self.dataLoader.numBatch * self.maxEpochs
  for i = 1, maxIter do
    self.encoderModel:zeroGradParameters()
    self.linkEmbeddingModel:zeroGradParameters()

    local data, pos_label, neg_label = unpack(self.dataLoader:nextBatch())

--    local encoder_data = self.encoderModel:forward(data)
    local pos_data = self.positiveEncoder:forward(data)
    local neg_data = self.negativeEncoder:forward(data)

    local pos_link = self.positiveLink:forward(pos_label)
    local neg_link = self.negativeLink:forward(neg_label)

    local pos_score = self.positiveDotProduct:forward({pos_data, pos_link})
    local neg_score = self.negativeDotProduct:forward({neg_data, neg_link})

    local loss =
      self.criterion:forward({pos_score, neg_score}, torch.Tensor(self.batchSize):fill(1))
    epochLoss = epochLoss + loss
    accumLoss = accumLoss + loss

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

    self.encoderModel:updateParameters(self.learningRate)
    self.linkEmbeddingModel:updateParameters(self.learningRate)

    if i % self.printEpoch == 0 then
      self.log.info(string.format("[Iter %d]: %f", i, accumLoss / self.printEpoch))
      accumLoss = 0
    end
    -- evaluate and save model
    if i % self.dataLoader.numBatch == 0 then
      local epoch = i / self.dataLoader.numBatch
      self.log.info(string.format("[Epoch %d]: %f", epoch, epochLoss / self.dataLoader.numBatch))
      --self:evaluate()
      --torch.save(self.modelDirectory.."/LSTM_"..epoch, self.biLSTM)
      epochLoss = 0
    end
  end
end

