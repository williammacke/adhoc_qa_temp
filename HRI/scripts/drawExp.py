
# Experiments
# Num Cols, Num Rows, Stations, Goal, Tool, Worker, Fetcher 

exp = [
        [   
            10,
            6,
            [[4,2], [4,4]],
            0,
            [[8,3], [8,3]],
            [0,3],
            [8,1]
        ],
        [   
            10,
            6,
            [[4,2], [4,4]],
            1,
            [[8,3], [8,3]],
            [0,3],
            [8,1]
        ],
        [   
            10,
            6,
            [[3,0], [7,0], [3,4], [7,4], [0,2], [5,0], [9,3], [4,5]],
            0,
            [[3,2], [3,2], [3,2], [3,2], [3,2], [3,2], [3,2], [3,2]],
            [5,2],
            [2,2]
        ],
        [   
            10,
            6,
            [[3,0], [7,0], [3,4], [7,4], [0,2], [5,1], [9,3], [4,5]],
            3,
            [[3,2], [3,2], [3,2], [3,2], [3,2], [3,2], [3,2], [3,2]],
            [5,2],
            [2,2]
        ],
        [   
            15,
            9,
            [[5,6], [8,3], [5,0], [2,3]],
            1,
            [[8,5], [8,5], [8,5], [8,5]],
            [5,3],
            [9,4]
        ],
        [   
            15,
            9,
            [[5,6], [8,3], [5,0], [2,3]],
            3,
            [[8,5], [8,5], [8,5], [8,5]],
            [5,3],
            [9,4]
        ],
        [   
            5,
            3,
            [[4,1], [0,1]],
            0,
            [[2,2], [2,2]],
            [2,1],
            [1,2]
        ],
        [   
            5,
            3,
            [[4,1], [0,1]],
            1,
            [[2,2], [2,2]],
            [2,1],
            [1,2]
        ],
        [   
            10,
            6,
            [[5,3], [5,2], [6,2]],
            0,
            [[8,3], [8,3], [8,3]],
            [0,3],
            [9,2]
        ],
        [   
            10,
            6,
            [[5,3], [5,2], [6,2]],
            2,
            [[8,3], [8,3], [8,3]],
            [0,3],
            [9,2]
        ],
    ]

directions = {"RIGHT" : (1, 0), 
              "LEFT"  : (-1, 0),
              "UP"    : (0, -1),
              "DOWN"  : (0, 1)}

from PIL import Image, ImageDraw
import sys

# Adjust large offset by 5, medium by 10, small by 10
large_offset  = 45
medium_offset = 90
small_offset  = 90
# Fill color adjust each time
color = (68, 35, 52)
# Change file name
experiment_file = open("../files/" + sys.argv[1] + "/" + sys.argv[3] + ".txt", 'r') # this is the file we want to read
exp_lines = experiment_file.readlines() 
exp_num = -1
  
# Go through each line in the experiment
exp_iter = iter(exp_lines)
line = next(exp_iter)
line = next(exp_iter)
try:
    while line: 
        if "EXPERIMENT" in line:
            exp_num = int(line[-2])
            print(exp_num)
            cur_exp = exp[exp_num]
            cur_pos = cur_exp[5]
            row = cur_exp[1]
            print("row", row)
            line = next(exp_iter)   

        with Image.open("../experiment_images/" + sys.argv[2] + "/ex" + str(exp_num) + ".PNG") as im:

            draw = ImageDraw.Draw(im)
            while "EXPERIMENT" not in line:
                line = line[0:15]
                for key in directions:
                    if key in line:
                        cur_dir = directions[key]
                        
                        if row == 6:
                            start = (cur_pos[0] * 125 + 35 + medium_offset, (row - cur_pos[1] - 1) * 160 + 65 + medium_offset)
                            end = (start[0] + (125 * cur_dir[0]), start[1] + (155 * cur_dir[1]))
                        elif row == 3:
                            start = (cur_pos[0] * 250 + 110 + small_offset, (row - cur_pos[1] - 1) * 320 + 140 + small_offset)
                            end = (start[0] + (240 * cur_dir[0]), start[1] + (320 * cur_dir[1]))
                        else: # row == 9
                            start = (cur_pos[0] * 80 + 45 + large_offset, (row - cur_pos[1] - 1) * 110 + 45 + large_offset)
                            end = (start[0] + (80 * cur_dir[0]), start[1] + (110 * cur_dir[1]))


                        x0, y0 = start
                        x1, y1 = end

                        # Now we can work out the x,y coordinates of the bottom of the arrowhead triangle
                        xb = 0.90*(x1-x0)+x0
                        yb = 0.90*(y1-y0)+y0

                        # Work out the other two vertices of the triangle
                        # Check if line is vertical
                        if x0==x1:
                           vtx0 = (xb-10, yb)
                           vtx1 = (xb+10, yb)

                        # Check if line is horizontal
                        else:
                           vtx0 = (xb, yb+10)
                           vtx1 = (xb, yb-10)

                        print(start, end)
                        draw.line([start, end], fill = color, width = 4)
                        draw.polygon([vtx0, vtx1, end], fill= color)

                        cur_pos[0] += cur_dir[0] 
                        cur_pos[1] -= cur_dir[1]
                line = next(exp_iter)

        im.save("../experiment_images/" + sys.argv[1] + "/ex" + str(exp_num) + ".PNG")
        
except:
    print('done')

im.save("../experiment_images/" + sys.argv[1] + "/ex" + str(exp_num) + ".PNG")

