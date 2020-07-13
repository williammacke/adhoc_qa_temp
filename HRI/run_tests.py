import os
import random
from zipfile import ZipFile

# Enable environment
# os.system("env/Scripts/activate")

# Randomize experiments
exp_num = [1, 2, 3, 4]
random.shuffle(exp_num)
for x in exp_num:
    # Run test
    os.system("py -3.6 HRI/human_experiment.py -i HRI/experiment"+ str(x) + ".txt -o output" + str(x) + ".txt")

# Zip up results
zipResults = ZipFile('results.zip', 'w')

for x in range(1,5):
    zipResults.write('output' + str(x) + '.txt') 
    os.remove('output' + str(x) + '.txt')
zipResults.close()