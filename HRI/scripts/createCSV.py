#1st argument is file with compiled data

import sys

if __name__ == "__main__":
    # Initialize data
    worker = ""
    results = {}
    numSteps = 0
    totalTime = 0.0
    exp = -1
    numWorkers = 0

    # Loop through all of the experiments
    compiledData = open(sys.argv[1], 'r')
    compiledLines = compiledData.readlines()
    for line in compiledLines:
        # Get worker id
        if "Worker: " in line: 
            recordData(exp, worker, totalTime, numSteps, results)
            worker = line[8:-1]
            exp = -1
            initResults(worker, results)
            numWorkers += 1
        
        # Get exdData(exp, worker, totalTime, numSteps, results)
            exp = int(line[-2:-1])
            numSteps = 0
            totalTimeperiment number and reset steps/time
        elif "EXPERIMENT" in line: 
            recor = 0.0

        # Add to the total time and steps
        elif "." in line: 
            totalTime += float(line[-16:-1].split()[0])
            numSteps += 1

    # Record last experiment
    recordData(exp, worker, totalTime, numSteps, results)

    displayResults(results, numWorkers)