from Bert_model4 import *
from data4 import *


model=BertModel(BertConfig())
model.build_model()
model.model.load_weights("bert_model4_epoch5acc0.9471428571428572")

data=train_data_generator()


with open("A14-恒生-测试数据集.txt","r",encoding="utf-8") as f:
	lines=f.readlines()

l=list(data.kb.keys())

dic={}
dic["team_name"]="小队"
dic["submit_result"]=[]

for i in lines:
	index,string=i.strip().split("\t")
	tmp={}
	tmp["text_id"]=int(index)
	tmp["text"]=string
	a,b,c=data.str2input(string)
	tmp["mention_result"]=[]
	res=model.model.predict(a)
	for j in range(len(res)):
		if(res[j]>0.43):
			index=l.index(b[j])
		else:
			index=-1
		tmp["mention_result"].append({"mention":b[j],"offset":c[j],"kb_id":index,"confidence":float(res[j][0])})
	dic["submit_result"].append(tmp)
        
with open("result.json","w",encoding="utf-8") as f:
		json.dump(dic,f,ensure_ascii=False,indent=1)
