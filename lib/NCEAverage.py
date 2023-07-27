import torch
from torch.autograd import Function
from torch import nn
from .alias_multinomial import AliasMethod
import math

class NCEFunction(Function):
    @staticmethod
    def forward(self, x, y, memory, idx, params):
        K = int(params[0].item())  # 特征x, 属于y是索引,  记忆特征memory, idx是抽样.
        T = params[1].item()
        Z = params[2].item()

        momentum = params[3].item()
        batchSize = x.size(0)
        outputSize = memory.size(0)
        inputSize = memory.size(1)

        # sample positives & negatives
        idx.select(1,0).copy_(y.data) # 取第一列数据.把y赋值过去. 我们知道y是真正的标签, idx是抽样的. 所以我们最后得到的idx数据是第一个数据是真实的, 后面4096个数据是假标签.其实提供的是论文中的分母均质化. 所以假标签里面是否存在真实标签无所谓. 

        # sample correspoinding weights
        weight = torch.index_select(memory, 0, idx.view(-1)) # 超大的一个memory矩阵, 保存 每个图片的自己特征.
        weight.resize_(batchSize, K+1, inputSize)

        # inner product  # 输入的x 跟字典里面数据都做一遍内机.
        out = torch.bmm(weight, x.data.resize_(batchSize, inputSize, 1))
        out.div_(T).exp_() # batchSize * self.K+1
        x.data.resize_(batchSize, inputSize)

        if Z < 0: # 归一化常数一定要大于0. 所以重新计算.
            params[2] = out.mean() * outputSize
            Z = params[2].item()
            print("normalization constant Z is set to {:.1f}".format(Z))

        out.div_(Z).resize_(batchSize, K+1)

        self.save_for_backward(x, memory, y, weight, out, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, weight, out, params = self.saved_tensors
        K = int(params[0].item())
        T = params[1].item()
        Z = params[2].item()
        momentum = params[3].item()
        batchSize = gradOutput.size(0)
        
        # gradients d Pm / d linear = exp(linear) / Z
        gradOutput.data.mul_(out.data)
        # add temperature
        gradOutput.data.div_(T)

        gradOutput.resize_(batchSize, 1, K+1) ##这个地方我跟源码区别是. 我删除了.data这几个字符, 看54行.不然又bug, 可能torch版本问题.
        
        # gradient of linear
        gradInput = torch.bmm(gradOutput.data, weight)
        gradInput.resize_as_(x)

        # update the non-parametric data
        weight_pos = weight.select(1, 0).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1-momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)
        
        return gradInput, None, None, None, None

class NCEAverage(nn.Module):

    def __init__(self, inputSize, outputSize, K, T=0.07, momentum=0.5, Z=None):
        super(NCEAverage, self).__init__()
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams) # 作用一个均匀分布的抽样器.
        self.multinomial
        self.K = K

        self.register_buffer('params',torch.tensor([K, T, -1, momentum]));
        stdv = 1. / math.sqrt(inputSize/3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2*stdv).add_(-stdv))
 
    def forward(self, x, y):
        batchSize = x.size(0)
        idx = self.multinomial.draw(batchSize * (self.K+1)).view(batchSize, -1) # 在一个均匀分布的抽样器里面我们抽取 128*4097个.
        out = NCEFunction.apply(x, y, self.memory, idx, self.params)
        return out

