import sys

#action that indicates legibility
legibility = {
    0: ("ANY", -1),
    1: ("ANY", 1),
    2: (1, "ANY"),
    3: (-1, "ANY"),
    4: [[-1, -1], [-2, -1], [-1, -2]],
    5: [[1, 1], [2, 1], [1, 2]],
    6: [[1, -1], [2, -1], [1, -2]],
    7: [[-1, 1], [-2, 1], [-1, 2]],
    8: (1, "ANY"),
    9: (-1, "ANY"),
    10: ("ANY", 1),
    11: ("ANY", -1),
    12: (1, "ANY"),
    13: (-1, "ANY"),
}

num_exp = 16
#start pos, size
exp_data = [
    ((0,3),"m"),
    ((0,3),"m"),
    ((4,1),"m"),
    ((4,1),"m"),
    ((5,2),"m"),
    ((5,2),"m"),
    ((5,2),"m"),
    ((5,2),"m"),
    ((5,3),"l"),
    ((5,3),"l"),
    ((5,3),"l"),
    ((5,3),"l"),
    ((2,1),"s"),
    ((2,1),"s"),
    ((0,3),"m"),
    ((0,3),"m")
]

grid_max = {
            "m": (9, 5),
            "s": (4, 2),
            "l": (14, 9),
            }

directions = {"RIGHT" : (1, 0), 
              "LEFT"  : (-1, 0),
              "UP"    : (0, 1),
              "DOWN"  : (0, -1)}


# timestep: numWorkers
timestepCount = {}

# workerId: legilibity count
legibilityCount = {}


def checkAdjustment(adjustment, exp_num):
    if (exp_num >= 0 and exp_num <= 3) or (exp_num >= 8 and exp_num <= 13):
        match = legibility[exp_num]
        if match[0] != "ANY":
            return adjustment[0] == match[0]
        if match[1] != "ANY":
            return adjustment[1] == match[1]

    
    if exp_num >= 4 and exp_num <= 7:
        match = legibility[exp_num]
        return adjustment in match

    return False

def read_lines(compiledData):
    compiledLines = compiledData.readlines()
    exp_num = -1
    for line in compiledLines:
        if "Worker ID: " in line: 
            worker = line[11:-1]
            exp = -1
            legibilityCount[worker] = 0

        if "EXPERIMENT" in line:
            exp_num = int(line[line.rfind('#') + 1:-1])
            adjustment = [0, 0]
            timestep = 0
            org_coor = exp_data[exp_num][0]

        elif '.' in line and timestep >= 0:
            line = line[0:15]
            for key in directions:
                if key in line:
                    adjustment[0] = adjustment[0] + directions[key][0]
                    adjustment[1] = adjustment[1] + directions[key][1]
                    next_coor = (org_coor[0] + adjustment[0], org_coor[1] + adjustment[1])
                    if next_coor[0] < 0 \
                        or next_coor[1] < 0 \
                        or next_coor[0] > grid_max[exp_data[exp_num][1]][0] \
                        or next_coor[1] > grid_max[exp_data[exp_num][1]][1]:

                        adjustment[0] = adjustment[0] - directions[key][0]
                        adjustment[1] = adjustment[1] - directions[key][1]

                    # Legible path
                    if checkAdjustment(adjustment, exp_num):
                        timestepCount[timestep] = 1 + (timestepCount[timestep] if timestep in timestepCount else 0)
                        timestep = -2
                        legibilityCount[worker] += 1

                    timestep = timestep + 1




if __name__ == "__main__":
    compiledData = open("../files/" + sys.argv[1] + "/compiled.txt", 'r')
    read_lines(compiledData)
    print(timestepCount)
    print(legibilityCount)