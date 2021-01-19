import sys
from PIL import Image, ImageDraw

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

def generate_coor_dict(compiledData):
    coor_array = [dict() for i in range(num_exp)]

    compiledLines = compiledData.readlines()
    exp_num = -1
    for line in compiledLines:
        if "EXPERIMENT" in line:
            exp_num = int(line[line.rfind('#') + 1:-1])
            coor_dict = coor_array[exp_num]
            cur_coor = exp_data[exp_num][0]

        elif '.' in line:
            line = line[0:15]
            for key in directions:
                if key in line:
                    next_coor = (cur_coor[0] + directions[key][0], cur_coor[1] + directions[key][1])
                    if next_coor[0] < 0 \
                        or next_coor[1] < 0 \
                        or next_coor[0] > grid_max[exp_data[exp_num][1]][0] \
                        or next_coor[1] > grid_max[exp_data[exp_num][1]][1]:
                        next_coor = cur_coor
                    coor = (cur_coor, next_coor)
                    coor_dict[coor] =  coor_dict[coor] + 1 if coor in coor_dict else 1

                    cur_coor = next_coor

    return coor_array

def draw_exp(coor_array):
    
    for exp_num in range(num_exp):
        with Image.open("../experiment_images/original/ex" + str(exp_num) + ".PNG") as im:
            draw = ImageDraw.Draw(im)
            for coords in coor_array[exp_num]:
                coords_val = coor_array[exp_num][coords]
                # color = (255, 0, 0)
                color = (int(255 * (10 - coords_val) / 10), int(255 * coords_val / 10), 0)
                size = exp_data[exp_num][1]
                # Line
                start = coords[0]
                end = coords[1]

                h_offset = 10 if start[0] < end[0] else 0
                v_offset = 10 if start[1] < end[1] else 0
                if size == "m":
                    start = (start[0] * 125 + 65 + v_offset, (6 - start[1] - 1) * 160 + 40 + h_offset)
                    end = (end[0] * 125 + 65 + v_offset, (6 - end[1] - 1) * 160 + 40 + h_offset)

                elif size == "s":
                    start = (start[0] * 250 + 110 + v_offset, (3 - start[1] - 1) * 320 + 140 + h_offset)
                    end = (end[0] * 250 + 110 + v_offset, (3 - end[1] - 1) * 320 + 140 + h_offset)

                elif size == "l":
                    start = (start[0] * 85 + 35 + v_offset, (9 - start[1] - 1) * 100 + 40 + h_offset)
                    end = (end[0] * 85 + 35 + v_offset, (9 - end[1] - 1) * 100 + 40 + h_offset)
                
                draw.line([start, end], fill = color, width = 6)

                # Arrow
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

                draw.polygon([vtx0, vtx1, end], fill= color)

        im.save("../experiment_images/" + sys.argv[1] + "/ex" + str(exp_num) + ".PNG")



if __name__ == "__main__":
    compiledData = open("../files/" + sys.argv[1] + "/compiled.txt", 'r')
    coor_array = generate_coor_dict(compiledData)
    draw_exp(coor_array)