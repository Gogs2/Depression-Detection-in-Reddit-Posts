import xml.etree.ElementTree as ET
import os
import pandas as pd

def getDataFromXML(path):
    tree = ET.parse(path)
    root = tree.getroot()
    ID=''
    posts = list()
    for elem in root:
        if elem.tag =='ID':
            ID = elem.text
        for subelem in elem:
            if subelem.tag == 'TEXT':
                    posts.append(subelem.text)

    return (ID,posts)

def getXmlFromPath(path):
    files = list()
    for r, d, f in os.walk(path):
        for file in f:
            if '.xml' in file:
                files.append(os.path.join(r, file))
    return files

def getData(files):
    data = dict()
    for f in files:
        id, text = getDataFromXML(f)
        if id not in data.keys():
            data[id]=text
        else: 
            data[id] = data[id] + text
    return data


