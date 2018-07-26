from nltk.parse.stanford import StanfordParser
from nltk import Tree
import json

parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def tree2dict(tree):
    return {tree.label(): [tree2dict(t) if isinstance(t, Tree) else t for t in tree]}
    

with open('targ-dev.txt') as f:
    lines = f.readlines()

output = {}
for index, line in enumerate(lines):
    if index > 100:
        break
    ls = tree2dict(list(parser.raw_parse(line))[0])
    output[index] = ls
    
with open('targ-dev.json', 'w') as outfile:
    json.dump(output, outfile)