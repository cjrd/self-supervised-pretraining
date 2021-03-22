#!/usr/bin/env python
"""
Quick script to convert the tensorflow output to OSS input files
"""
import sys
import re

inp = sys.argv[1]
class_rgx = re.compile('([^\d]+)\d+.tif')
classes = set()
with open(inp) as f:
    for line in f:
        line=line.strip()
        res=class_rgx.match(line)
        classes.add(res[1])
classes=sorted(classes)


with open(inp) as f:
    for line in f:
        line=line.strip()
        res=class_rgx.match(line)
        print("Images/{}/{} {}".format(res[1], line, classes.index(res[1])))

