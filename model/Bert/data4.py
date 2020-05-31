import json
import pandas
from utils import OurTokenizer
import random
import numpy

TRAIN_DATA_PATH = r"./train_dev/train.json"
KB_PATH = r"./knowledge_base.txt"
DICT_PATH = r'./vocab.txt' #bert字典的位置
replace_dict={"“":"\"",
              "”":"\""}
info=pandas.read_csv("./train_dev/company_2_code_full.txt",dtype=object,sep="	",header=None,names=["stock_name","stock_full_name","stock_code"])

def replace_sub_to_full(sentence,entity):
    for i in info.iterrows():
        if(i[1]["stock_name"]==entity):
            return sentence.replace(entity,i[1]["stock_full_name"])


class train_data_generator():
    def __init__(self,test_size=700,batch_size=64,seed=123):
##        self.rand=random.Random(seed)
        #载入数据集
        with open(TRAIN_DATA_PATH,"r",encoding='utf8') as f:
            data=json.load(f)
##        self.rand.shuffle(data)
        self.train_data=data[:-test_size]
        self.test_data=data[-test_size:]
        self.batch_size=batch_size
        if len(self.train_data)%batch_size==0:
            self.train_step=len(self.train_data)//batch_size
        else:
            self.train_step=len(self.train_data)//batch_size+1
            
        if len(self.test_data)%batch_size==0:
            self.test_step=len(self.test_data)//batch_size
        else:
            self.test_step=len(self.test_data)//batch_size+1
        
        #载入知识库
        with open(KB_PATH,"r",encoding='utf8') as f:
            self.kb=json.load(f)
        token_dict = {}
        #载入bert词典
        with open(DICT_PATH, 'r', encoding='utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        self.tokenizer = OurTokenizer(token_dict)
        #构建字典树
        self.init_trie_tree()

        

    def init_trie_tree(self):
        info=pandas.read_csv("./train_dev/company_2_code_full.txt",dtype=object,sep="	",header=None,names=["stock_name","stock_full_name","stock_code"])
        dic={}
        for i in info.iterrows():
            tmp=dic
            for j in i[1]["stock_name"]:
                if j in tmp.keys():
                    tmp=tmp[j]
                else:
                    tmp[j]={}
                    tmp=tmp[j]
            if "stop" in tmp.keys():
                tmp["stop"].append({"stock_name":i[1]["stock_name"],"stock_full_name":i[1]["stock_full_name"]})
            else:
                tmp["stop"]=[]
                tmp["stop"].append({"stock_name":i[1]["stock_name"],"stock_full_name":i[1]["stock_full_name"]})
        self.dic=dic
    
    def _process(self,text):   #字符替换
        for i in replace_dict.items():
            text=text.replace(i[0],i[1])
        return text.lower()

    def seq_padding(self,X, padding=0):  #输入长度对齐
##        return numpy.array(X)
        L = [len(x) for x in X]
        ML = max(L)
        return numpy.array([
            numpy.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
        ])
    
    def train_batch(self):
        count=0
        while(True):
            x1=[]
            x2=[]
            masks=[]
            x3=[]
            x4=[]
            mask1=[]
            x5=[]
            x6=[]
            mask2=[]
            y=[]
            for i in range(self.batch_size):
                tmp=self.train_data[count]["text"]+"。"+self.kb[self.train_data[count]["lab_result"][0]["mention"]]
                tmp=self._process(tmp)
                tmp=self.tokenizer.encode(tmp)
                
                x1.append(tmp[0])
                x2.append(tmp[1])
                mask=numpy.array([False for i in range(len(tmp[0]))])
                mask[0]=True
                mask[self.train_data[count]["lab_result"][0]["offset"]+1]=True 
                mask[self.train_data[count]["lab_result"][0]["offset"]+len(self.train_data[count]["lab_result"][0]["mention"])]=True
                assert len(self.train_data[count]["lab_result"][0]["mention"])>1
                masks.append(mask)
                
                tmp=self.train_data[count]["text"].split("，")
                for i in tmp:
                    if(i.find(self.train_data[count]["lab_result"][0]["mention"])!=-1):
                        for j in i.split("。"):
                            if(j.find(self.train_data[count]["lab_result"][0]["mention"])!=-1):
                                t1=self.tokenizer.encode(self._process(j))
                                x3.append(t1[0])
                                x4.append(t1[1])
                                m1=numpy.array([False for i in range(len(t1[0]))])
                                m1[0]=True
                                mask1.append(m1)

                                t2=replace_sub_to_full(j,self.train_data[count]["lab_result"][0]["mention"])
                                t2=self.tokenizer.encode(self._process(t2))
                                x5.append(t2[0])
                                x6.append(t2[1])
                                m2=numpy.array([False for i in range(len(t2[0]))])
                                m2[0]=True
                                mask2.append(m2)
                                break
                        break
                if self.train_data[count]["lab_result"][0]["kb_id"]==-1:
                    y.append(0)
                else:
                    y.append(1)
                count+=1
                if count==len(self.train_data):
                    count=0
                    break
            x1=self.seq_padding(x1)
            x2=self.seq_padding(x2)
            masks=self.seq_padding(masks,padding=False)

            x3=self.seq_padding(x3)
            x4=self.seq_padding(x4)
            mask1=self.seq_padding(mask1,padding=False)

            x5=self.seq_padding(x5)
            x6=self.seq_padding(x6)
            mask2=self.seq_padding(mask2,padding=False)
            
            yield [x1,x2,masks,x3,x4,mask1,x5,x6,mask2],y

    def str2input(self,string):
        x1=[]
        x2=[]
        masks=[]
        x3=[]
        x4=[]
        mask1=[]
        x5=[]
        x6=[]
        mask2=[]
        
        entity=[]
        off=[]
        for i in range(len(string)):
            tmp=self.dic
            offset=0
            while(i+offset<len(string) and string[i+offset] in tmp.keys()):
                tmp=tmp[string[i+offset]]
                offset+=1
                if("stop" in tmp.keys()):
                    for j in tmp["stop"]:
                        _tmp=string+"。"+self.kb[j["stock_name"]]
                        _tmp=self._process(_tmp)
                        if(len(_tmp)>510):  #句子过长时截断
                            
                            _tmp=_tmp[0:510]
                        print(len(_tmp))
                        _tmp=self.tokenizer.encode(_tmp)
                        x1.append(_tmp[0])
                        x2.append(_tmp[1])
                        mask=numpy.array([False for i in range(len(_tmp[0]))])
                        mask[0]=True
                        mask[i+1]=True 
                        mask[i+offset]=True
                        masks.append(mask)

                        tmpstr=string.split("，")
                        for k in tmpstr:
                            if(k.find(j["stock_name"])!=-1):
                                for b in k.split("。"):
                                    if(b.find(j["stock_name"])!=-1):
                                        t1=self.tokenizer.encode(self._process(b))
                                        x3.append(t1[0])
                                        x4.append(t1[1])
                                        m1=numpy.array([False for i in range(len(t1[0]))])
                                        m1[0]=True
                                        mask1.append(m1)

                                        t2=replace_sub_to_full(b,j["stock_name"])
                                        t2=self.tokenizer.encode(self._process(t2))
                                        x5.append(t2[0])
                                        x6.append(t2[1])
                                        m2=numpy.array([False for i in range(len(t2[0]))])
                                        m2[0]=True
                                        mask2.append(m2)
                                        break
                                break
                        
                        off.append(i)
                        entity.append(j["stock_name"])
        x1=self.seq_padding(x1)
        x2=self.seq_padding(x2)
        masks=self.seq_padding(masks,padding=False)

        x3=self.seq_padding(x3)
        x4=self.seq_padding(x4)
        mask1=self.seq_padding(mask1,padding=False)

        x5=self.seq_padding(x5)
        x6=self.seq_padding(x6)
        mask2=self.seq_padding(mask2,padding=False)
        return [x1,x2,masks,x3,x4,mask1,x5,x6,mask2],entity,off

    def test_batch(self):
        count=0
        while(True):
            x1=[]
            x2=[]
            masks=[]
            x3=[]
            x4=[]
            mask1=[]
            x5=[]
            x6=[]
            mask2=[]
            y=[]
            for i in range(self.batch_size):
                tmp=self.test_data[count]["text"]+"。"+self.kb[self.test_data[count]["lab_result"][0]["mention"]]
                tmp=self._process(tmp)[:510]
                tmp=self.tokenizer.encode(tmp)
                
                x1.append(tmp[0])
                x2.append(tmp[1])
                mask=numpy.array([False for i in range(len(tmp[0]))])
                mask[0]=True
                mask[self.test_data[count]["lab_result"][0]["offset"]+1]=True
                mask[self.test_data[count]["lab_result"][0]["offset"]+len(self.test_data[count]["lab_result"][0]["mention"])]=True

                assert len(self.test_data[count]["lab_result"][0]["mention"])>1
                masks.append(mask)
    
                tmp=self.test_data[count]["text"].split("，")
                for i in tmp:
                    if(i.find(self.test_data[count]["lab_result"][0]["mention"])!=-1):
                        for j in i.split("。"):
                            if(j.find(self.test_data[count]["lab_result"][0]["mention"])!=-1):
                                t1=self.tokenizer.encode(self._process(j))
                                x3.append(t1[0])
                                x4.append(t1[1])
                                m1=numpy.array([False for i in range(len(t1[0]))])
                                m1[0]=True
                                mask1.append(m1)
        
                                t2=replace_sub_to_full(j,self.test_data[count]["lab_result"][0]["mention"])
                                t2=self.tokenizer.encode(self._process(t2))
                                x5.append(t2[0])
                                x6.append(t2[1])
                                m2=numpy.array([False for i in range(len(t2[0]))])
                                m2[0]=True
                                mask2.append(m2)
                                break
                        break

                
                if self.test_data[count]["lab_result"][0]["kb_id"]==-1:
                    y.append(0)
                else:
                    y.append(1)
                count+=1
                if count==len(self.test_data):
                    count=0
                    break
            x1=self.seq_padding(x1)
            x2=self.seq_padding(x2)
            masks=self.seq_padding(masks,padding=False)

            x3=self.seq_padding(x3)
            x4=self.seq_padding(x4)
            mask1=self.seq_padding(mask1,padding=False)

            x5=self.seq_padding(x5)
            x6=self.seq_padding(x6)
            mask2=self.seq_padding(mask2,padding=False)
            
            yield [x1,x2,masks,x3,x4,mask1,x5,x6,mask2],y
