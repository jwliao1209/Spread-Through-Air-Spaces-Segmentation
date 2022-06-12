import csv
import json
import yaml

from .Utils import cfgNode_to_dict

__all__ = ["ReadTxt", "ReadCsv", "ReadJson", "ReadYaml",
           "WriteTxt", "WriteCsv", "WriteJson", "WriteYaml"]

# ============== .txt file related =============
def WriteTxt(txtLoc, content):
    with open(txtLoc, 'w') as fp:
        for line in content:
            fp.write(line+'\n')
    return

def ReadTxt(txtLoc):
    with open(txtLoc, 'r') as fp:
        content = fp.readlines()
        content = [line.strip() for line in content]
    return content

# ============== .csv file related =============
def WriteCsv(csvLoc, titleRow, contentRows):
    with open(csvLoc, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=titleRow)
        writer.writeheader()
        for line in contentRows:
            writer.writerow(line)

def ReadCsv(csvLoc):
    with open(csvLoc, newline='') as csvfile:
        content = list(csv.reader(csvfile))
        fields, content = content[0], content[1:]
        contentDict = dict(zip(fields, zip(*content)))
    return contentDict

# ============= .json file related ================
def WriteJson(jsonLoc, content):
    with open(jsonLoc, 'w', newline='\n') as jsonfile:
        json.dump(content, jsonfile, indent=4)
    return

def ReadJson(jsonLoc):
    with open(jsonLoc, newline='') as jsonfile:
        data = json.load(jsonfile)
    return data

# ============== .yaml related ===================
def ReadYaml(yamlloc):
    with open(yamlloc, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

def WriteYaml(yamlLoc, CfgNode):
    CfgDict = cfgNode_to_dict(CfgNode.clone())
    with open(yamlLoc, 'w') as yamlfile:
        yaml.dump(CfgDict, yamlfile)
    return
