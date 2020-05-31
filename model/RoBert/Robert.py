import torch
from transformers import *
import torch.nn.functional as F
from data4 import train_data_generator

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.RoBert=BertModel.from_pretrained("../aibert")
        for p in self.parameters():
            p.requires_grad = False
        self.hidden=torch.nn.Linear(768*3+1,256)
        self.res=torch.nn.Linear(256,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        out=self.RoBert(torch.LongTensor(x[0]).cuda())[0]
        out1=self.RoBert(torch.LongTensor(x[1]).cuda())[0]
        out2=self.RoBert(torch.LongTensor(x[2]).cuda())[0]
        x=torch.masked_select(out,torch.BoolTensor(x[3]).cuda())
        x=x.reshape([-1,768*3])
        tensor1=torch.index_select(out1,1,torch.LongTensor([0]).cuda())
        tensor1=torch.reshape(tensor1,[64,768])
        tensor2=torch.index_select(out2,1,torch.LongTensor([0]).cuda())
        tensor2=torch.reshape(tensor1,[64,768])
        sim=torch.cosine_similarity(tensor1,tensor2).reshape([-1,1])
        concat=torch.cat((x,sim),1)
        x=F.relu(self.hidden(concat))
        x=F.relu(self.res(x))
        x=self.sigmoid(x)
        return x

##if __name__=='__main__':
##    model=MyModel()
##    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
##    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
##    data=train_data_generator()
##    train=data.train_batch()
##    model.cuda()
##    for i in range(data.train_step):
##        optimizer.zero_grad() 
##        x,y=next(train)
##        out=model(x)
##        criterion = torch.nn.BCELoss()
##        out=out.reshape([64])
##        loss=criterion(out,torch.Tensor(y).cuda())
##        loss.backward()
##        optimizer.step()
##        print("step:"+str(i)+str(torch.mean(loss)))
##    
