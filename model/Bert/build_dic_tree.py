import json
import pandas

##info=pandas.read_csv("./train_dev/company_2_code_sub.txt",dtype=object,sep="	")
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

def scan(string):
    assert isinstance(string,str)
    res=[]
    off=[]
    for i in range(len(string)):
        tmp=dic
        offset=0
        while(i+offset<len(string) and string[i+offset] in tmp.keys()):
            tmp=tmp[string[i+offset]]
            offset+=1
            if("stop" in tmp.keys()):
                for j in tmp["stop"]:
                    res.append(j)
                    off.append(i)
    return (res,off)
