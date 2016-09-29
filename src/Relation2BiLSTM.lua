require 'rnn'
require 'DataLoader'
require 'utils'
require 'logroll'

local Relation2BiLSTM = torch.class('Relation2BiLSTM')

function Relation2BiLSTM:__init(opt)
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
  self.validDataLoader
    = DataLoader(self.validDataPath, 1, self.vocabularySize, self.relationSize)


  self.linkEmbedding = nn.LookupTable(self.relationSize, self.relationDim)
  self.wordEmbedding = nn.LookupTable(self.vocabularySize, self.vocabularyDim)


  self.firstlayerForward = nn.Sequential()
            :add(nn.Sequencer(self.wordEmbedding))
            :add(nn.Sequencer(nn.LSTM(self.hiddenSize, self.hiddenSize)))
  self.firstlayerBackward = nn.Sequential()
            :add(nn.ReverseTable())
            :add(nn.Sequencer(self.wordEmbedding))
            :add(nn.Sequencer(nn.LSTM(self.hiddenSize, self.hiddenSize)))
  self.firstlayerConcat = nn.ConcatTable()
            :add(self.firstlayerForward):add(self.firstlayerBackward)

  self.secondlayerForward = nn.Sequential()
            :add(nn.Sequencer(nn.LSTM(self.hiddenSize * 2, self.hiddenSize)))
            :add(nn.SelectTable(-1))
  self.secondlayerBackward = nn.Sequential()
            :add(nn.ReverseTable())
            :add(nn.Sequencer(nn.LSTM(self.hiddenSize * 2, self.hiddenSize)))
            :add(nn.SelectTable(-1))
  self.secondlayerConcat = nn.ConcatTable()
            :add(self.secondlayerForward):add(self.secondlayerBackward)
  
  self.encoder = nn.Sequential()
            :add(self.firstlayerConcat)
            :add(nn.ZipTable())
            :add(nn.Sequencer(nn.JoinTable(1, 1)))
            :add(self.secondlayerConcat)
            :add(nn.JoinTable(1, 1))
            :add(nn.Linear(self.hiddenSize * 2, self.relationDim))

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

function Relation2BiLSTM:train()
  local epochLoss, accumLoss = 0, 0
  local maxIter = self.dataLoader.numBatch * self.maxEpochs
  for i = 1, maxIter do
    xlua.progress(i, maxIter)

    self.encoderModel:zeroGradParameters()
    self.linkEmbeddingModel:zeroGradParameters()

    local data, pos_label, neg_label = unpack(self.dataLoader:nextBatch())
    local inputSeq = data:t():split(1, 1)
    for j = 1, #inputSeq, 1 do
      inputSeq[j] = torch.squeeze(inputSeq[j])
    end

    local pos_data = self.positiveEncoder:forward(inputSeq)
    local neg_data = self.negativeEncoder:forward(inputSeq)

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
    self.positiveEncoder:backward(inputSeq, dev_pos_data)
    self.negativeLink:backward(neg_label, dev_neg_link)
    self.negativeEncoder:backward(inputSeq, dev_neg_data)

    self.encoderModel:updateParameters(self.learningRate)
    self.linkEmbeddingModel:updateParameters(self.learningRate)

    if i % self.printEpoch == 0 then
      self.log.info(string.format("[Iter %d]: %f", i, accumLoss / self.printEpoch))
      accumLoss = 0
    end
    -- evaluate and save model
    if i % self.dataLoader.numBatch == 0 then
      local epoch = i / self.dataLoader.numBatch
      local position = self:evaluate()
      self.log.info(
        string.format("[Epoch %d]: [training error %f]: [evaluating error %d]",
        epoch, epochLoss / self.dataLoader.numBatch, position))
      --torch.save(self.modelDirectory.."/LSTM_"..epoch, self.biLSTM)
      chLoss = 0
    end
  end
end

function Relation2BiLSTM:evaluate()
  local relationsID = torch.Tensor(self.relationSize)
  local s = relationsID:storage()
  for i = 1, s:size() do
    s[i] = i
  end
  local relationsVector = self.linkEmbeddingModel:forward(relationsID)

  local position = 0
  local eDotProduct = cudacheck(nn.DotProduct())
  for i = 1, self.validDataLoader.dataSize do
    local data, label, _ = unpack(self.validDataLoader:nextBatch())

    local inputSeq = data:t():split(1, 1)
    for j = 1, #inputSeq, 1 do
      inputSeq[j] = torch.LongTensor(1):fill(torch.squeeze(inputSeq[j]))
    end
    self.encoderModel:zeroGradParameters()
    self.linkEmbeddingModel:zeroGradParameters()
    local dataVector = torch.repeatTensor(self.encoderModel:forward(inputSeq), self.relationSize, 1)
    local scoreVector = eDotProduct:forward({dataVector, relationsVector})

    position = position + torch.gt(scoreVector, scoreVector[label[1]]):sum() / self.validDataLoader.dataSize
  end
  return position
end

