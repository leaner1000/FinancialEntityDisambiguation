import scrapy
import pandas

from stock_kb.items import StockKbItem

class RoyalflushSpider(scrapy.spiders.Spider):
    name = "Royalflush"
    allowed_domains = ["stockpage.10jqka.com.cn"]
    info=pandas.read_csv("./company_2_code_full.txt",dtype=object,sep="	",header=None,names=["stock_name","stock_full_name","stock_code"])
    start_urls = ["http://stockpage.10jqka.com.cn/"+i+"/company.html#stockpage" for i in info["stock_code"]]

    def parse(self, response):
        item = StockKbItem()
        item["stock_id"]=response.url.split("/")[3]
        for i in response.xpath('//table[@class="m_table"]//td'):
            if(len(i.xpath('.//strong/text()'))>0 and i.xpath('.//strong/text()').extract()[0]=='公司名称：'):
                item["stock_full_name"]=i.xpath('.//span/text()').extract()[0]

        for i in response.xpath('//div[@class="m_tab_content2"]//table//tr'):
            if(len(i.xpath('.//strong/text()'))>0 and i.xpath('.//strong/text()').extract()[0]=='主营业务：'):
                item["major_bussiness"]=i.xpath('.//span/text()').extract()[0]
            if(len(i.xpath('.//strong/text()'))>0 and i.xpath('.//strong/text()').extract()[0]=='办公地址：'):
                item["location"]=i.xpath('.//span/text()').extract()[0]
            if(len(i.xpath('.//strong/text()'))>0 and i.xpath('.//strong/text()').extract()[0]=='公司简介：'):
                item["description"]=i.xpath('.//p[@class="tip lh24"]/text()').extract()[0]
        print(item["stock_full_name"])
        return item
