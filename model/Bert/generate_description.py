import pandas
import json

info=pandas.read_csv("./train_dev/company_2_code_full.txt",dtype=object,sep="	",header=None,names=["stock_name","stock_full_name","stock_code"])
with open("./items.json","r",encoding="utf-8") as f:
	kb=json.load(f)

dic={}
description={}

for i in kb:
	dic[i["stock_id"]]=i

for j,i in info.iterrows():
	if(i["stock_code"] not in dic.keys()):
		print(i["stock_code"])
		continue
	string="代码："+dic[i["stock_code"]]["stock_id"]+"。"
	string=string+"全称："+dic[i["stock_code"]]["stock_full_name"]+"。"
	string=string+"主营业务："+dic[i["stock_code"]]["major_bussiness"]+"。"
	string=string+"办公地点："+dic[i["stock_code"]]["location"]+"。"
	string=string+"简介："+dic[i["stock_code"]]["description"]+"。"
	description[i["stock_name"]]=string

dic["华凯实业"]='代码：400007。全称：海南华凯实业股份有限公司。'
dic["湘农股份"]='代码：400009。全称：广东湘农绿色农业股份有限公司。'
dic["港岳航电"]='代码：400013。全称：山东港岳航电集团股份有限公司。'

with open("description.json","w",encoding="utf-8") as f:
	json.dump(description,f,ensure_ascii=False,indent=1)
