from Bert_model4 import *
from data4 import *
config=BertConfig()
model=BertModel(config)
model.build_model()
    
data=train_data_generator()
train=data.train_batch()
test=data.test_batch()

f1_max=0
acc_max=0
f1_max_step=0
acc_max_step=0
threshold=0.5
l=[]
model.model.load_weights("bert_model4_epoch5acc0.9471428571428572")
res=[]
ans=[]
_eval=[]
for j in range(data.test_step):
    x,y=next(test)
    r=model.model.predict(x)
    for j in range(len(r)):
        res.append(r[j][0])
        ans.append(y[j])

for j in range(99):
    threshold=j*0.01
    tp=0
    fp=0
    tn=0
    fn=0
    for k in range(len(res)):
        if res[k]>threshold:
            if ans[k]==1:
                tp+=1
            if ans[k]==0:
                fp+=1
        if res[k]<threshold:
            if ans[k]==0:
                tn+=1
            if ans[k]==1:
                fn+=1
        if(tp==0 or fp==0 or fn==0 or tn==0):
            continue
        pre=tp/(tp+fp)
        recall=tp/(tp+fn)
        acc=(tp+tn)/700
        fpr=fp/(fp+tn)
        tpr=tp/(tp+fn)
        f1=2*pre*recall/(pre+recall)
        _eval.append([tp,fp,tn,fn])
        if(acc_max<acc):
            acc_max=acc
        if(f1_max<f1):
            f1_max=f1
    if(tp==0 or fp==0 or fn==0 or tn==0):
            continue
    print("threshold:{0} acc:{1} recall:{2} pre:{3} f1:{4}".format(threshold,acc,recall,pre,f1))
    l.append(_eval)

