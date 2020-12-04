# 1st argument is filenames file
# 2nd argument is batch file
# 3rd argument is output file

import csv
import sys

keys = open(sys.argv[1], 'r') # filenames.txt is the list of files
Lines = keys.readlines() 
  
# Strips the newline character 
validKeys = []
for line in Lines: 
    validKeys.append(line.strip()[:-4])

#temp.csv is resulting file
with open(sys.argv[3], 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    # Batch file is output of MTurk
    with open(sys.argv[2], newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        first = True
        surveyCol = -1
        approveCol = -1
        rejectCol = -1

        for row in reader:
            if first:
                count = 0
                for header in row:
                    if header == "Answer.surveycode":
                        surveyCol = count
                    elif header == "Approve":
                        approveCol = count
                    elif header == "Reject":
                        rejectCol = count
                    count += 1
                first = False
            else:
                if row[surveyCol].replace('"', '') in validKeys:
                    row.append("x")
                else:
                    row.append("")
                    row.append("Not a valid key")

            writer.writerow(row)
        