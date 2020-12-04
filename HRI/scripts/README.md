
## Windows

To write all filenames to a text file:

```
dir /b > filenames.txt
```

## checkKeys.py

```
python filenames.txt Batch_1233456.csv output.csv
```

## drawExp.py

At this line of code

```
    im.save("../experiment_images/11-9/ex" + str(exp_num) + ".PNG")
```

replace the date with the correct date.

At this line of code

```
with Image.open("../experiment_images/11-9/ex" + str(exp_num) + ".PNG") as im:
```

replace the date with "original" for the first run. After the first run, replace it with the folder name of the date.

For each run you will need to modify these lines by the specified comments:
```
# Adjust large offset by 5, medium by 10, small by 10
large_offset = 45
medium_offset = 90
small_offset = 90
# Fill color adjust each time
color = (68, 35, 52)
# Change file name
experiment_file = open('../files/11-9/workerKey.txt', 'r') # this is the file we want to read
```

## compileData.py

From within scripts/ run this command
The first argument is the folder name containing all of the data and should be located in "files"
The second argument is the name of the output file you want to write to
```
python compileData.py 11-9 output.txt
```

## analyzeResults.py

From within scripts/ run this command
The first argument is the folder name containing all of the data and should be located in "files"
The second argument is the name of the output file you want to write to
```
python analyzeResults.py output.txt
```