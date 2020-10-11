#!/usr/bin/env python

import sys
import json
import cgi

fs = cgi.FieldStorage() # POST data

d = {}
for k in fs.keys():
    d[k] = fs.getvalue(k)

filename = d['filename']
text = d['data']

output = open(filename, "a")

if(text == "date"):

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    output.write(dt_string + "\n")
    output.write("{0:15} {1:15} {2:15}\n".format("WORKER ACTION", "FETCHER ACTION", "TIME ELAPSED"))


else:
    output.write(text)