#1st argument is file with compiled data

import sys

def recordData(exp, worker, totalTime, numSteps, results):
    if exp != -1:
        results[worker]["avgTimes"][exp] = totalTime / numSteps
        results[worker]["steps"][exp] = numSteps

def initResults(worker, results):
    results[worker] = {}
    results[worker]["avgTimes"] = [0] * 10
    results[worker]["steps"] = [0] * 10

def displayResults(results, numWorkers):
    # numTopWorkers = numWorkers // 10
    numTopWorkers = 2
    workers = list(results)
    steps = []
    expAvgTimes = [0.0] * 10

    for worker in workers:
        # Get total number of steps for this worker
        steps.append(sum(results[worker]["steps"]))

        # Add average times to total average times by experiment
        avgTimes = results[worker]["avgTimes"]
        for i in range(len(avgTimes)):
            expAvgTimes[i] += avgTimes[i]
    
    # Get top workers based on the number of steps
    topWorkersSteps = sorted(steps)[:numTopWorkers]
    
    # Calculate averages of the experiments
    expAvgTimes = list(map(lambda x: x / numWorkers, expAvgTimes))

    # Get data for top workers
    topWorkerKeys = []
    topWorkerTimes = {}
    for workerSteps in topWorkersSteps:
        workerKey = workers[steps.index(workerSteps)]
        topWorkerKeys.append(workerKey)
        topWorkerTimes[workerKey] = results[workerKey]["avgTimes"]

    print("-----------------------")
    
    # Print Average Times for each experiment
    print ("\nAverage Times for each experiment")
    for i in range(len(expAvgTimes)):
        print("{:>16s}{:>15f}".format("Experiment " + str(i), expAvgTimes[i]))
    print("{:>19s}{:>12f}".format("Overall Average", sum(expAvgTimes) / 10))

    # Print the keys of the top workers
    print("\nTop Worker Keys (for least amount of steps)")
    for worker in topWorkerKeys:
        print("{:>20s}".format(worker))
    
    # Print the average times for the experiments for top workers
    print("\nTop Worker Average Times")
    for worker in topWorkerKeys:
        print("\n{:>20s}".format(worker))
        workerTimes = topWorkerTimes[worker]
        for i in range(len(workerTimes)):
            print("{:>20s}{:>15f}".format("Experiment " + str(i), workerTimes[i]))
        print("{:>23s}{:>12f}".format("Overall Average", sum(workerTimes) / 10))
    print("-----------------------")


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
        
        # Get experiment number and reset steps/time
        elif "EXPERIMENT" in line: 
            recordData(exp, worker, totalTime, numSteps, results)
            exp = int(line[-2:-1])
            numSteps = 0
            totalTime = 0.0

        # Add to the total time and steps
        elif "." in line: 
            totalTime += float(line[-16:-1].split()[0])
            numSteps += 1

    # Record last experiment
    recordData(exp, worker, totalTime, numSteps, results)

    displayResults(results, numWorkers)
