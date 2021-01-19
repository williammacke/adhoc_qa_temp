
## Windows

To write all filenames to a text file:

```
dir /b > filenames.txt
```

## checkKeys.py

```
python checkKeys.py filenames.txt Batch_1233456.csv output.csv
```

## compileData.py

This script compiles all of the worker data into one file.

From within scripts/ run this command
The first argument is the folder name containing all of the data and should be located in "files"
The second argument is the name of the output file you want to write to
```
python compileData.py 11-9 output.txt
```


## analyzeResults.py

This script gets the top 10% of workers, gets total average time etc.
A new csv will also be generated with the average times for each worker

From within scripts/ run this command
The first argument is the output file from compileData.py
```
python analyzeResults.py output.txt output.csv
```

## drawExp2.py

This is one script that will use the compiled data (should be called compiled.txt and within the files/date/ folder)
to generate an image of the experiment. 

```
python drawExp2.py 1-15
```


## drawExp.py (OLD)

For first worker: 

```
python drawExp.py 12-7 original workerID
```

For all workers after:

```
python drawExp.py 12-7 12-7 workerID
```

For each run you will need to modify these lines by the specified comments:
```
# Adjust large offset by 5, medium by 10, small by 10
large_offset = 5
medium_offset = 10
small_offset = 10
# Fill color adjust each time
color = (23, 18, 25)
```