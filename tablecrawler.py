"""This module provides a sample implentation crawling a webpage table into tables."""

import json
import logging
from bs4 import BeautifulSoup
import requests

def get_tables(*args):
    """get webpage table into list"""
    url, id, value = args
    #fake useragent in http request headers
    httpheaders = {'User-Agent': 
                    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    html=requests.get(url, headers=httpheaders)
   
    if __debug__:
        if not html.status_code == 200: raise AssertionError

    soup = BeautifulSoup(html.text,"lxml")
    logging.debug(soup)

    #get table by ID
    table = soup.find(lambda tag: tag.name=='table' and tag.has_attr(id) and tag[id]==value)
 
    #looks stupid
    data = list()
    rows = list()
    cols = list()
    table_headers = list()
    for tx in table.find_all('th'):
        table_headers.append(tx.text.strip()) 
    data.append(table_headers)
    rows=table.find_all('tr')
    for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols if ele]
            data.append([ele for ele in cols if ele]) # Get rid of empty values
    
    return data

def export_json(data):
    """Export list to json format"""
    json_dict = {}
    heading = data[0]
    for row in data [1:]:
        for col_header, data_colume in zip(heading,row): 
            json_dict.setdefault(col_header,[]).append(data_colume) 
    return json.dumps(json_dict)

def traverse_list(data):
    """traverse list"""
    cells = list()
    for cells in data:
        print (cells)
