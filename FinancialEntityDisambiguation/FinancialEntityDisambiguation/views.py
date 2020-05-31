from django.http import HttpResponse
import json


with open(r"D:\桌面\临时2\服创大赛\knowledge_base.txt","r",encoding='utf8') as f:
    kb=json.load(f)

l=list(kb.keys())

def entity_description(request):
    kb_id=int(request.GET.get("kb_id"))
    kb_name=l[kb_id]
    return HttpResponse(kb[kb_name])
