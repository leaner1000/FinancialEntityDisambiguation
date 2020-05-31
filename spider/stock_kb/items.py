# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class StockKbItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    stock_id=scrapy.Field()
    stock_full_name=scrapy.Field()
    major_bussiness=scrapy.Field()
    description=scrapy.Field()
    location=scrapy.Field()
