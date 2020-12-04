#1st argument is input
#2nd argument is output

import os.path
import sys

output = open(sys.argv[2], 'w')

for filename in os.listdir('../files/' + sys.argv[1] +'/'):
    fileData = open('../files/' + sys.argv[1] +'/' + filename, 'r')
    lines = fileData.readlines() 
    workerId = filename[:-4]
    output.write("Worker ID: " + workerId + "\n")
    for line in lines:
        output.write(line)
