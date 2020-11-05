import csv

keys = open('temp.txt', 'r') 
Lines = keys.readlines() 
  
# Strips the newline character 
validKeys = []
for line in Lines: 
    validKeys.append(line.strip()[:-4])

with open('temp.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    with open('Batch_4222921_batch_results.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        first = True
        surveyCol = -1
        approveCol = -1
        rejectCol = -1

        for row in reader:
            if first:
                count = 0
                for header in row:
                    print(header)
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
        