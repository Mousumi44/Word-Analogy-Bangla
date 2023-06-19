import re
import subprocess


filename = "Tense"

# Define a regular expression pattern to match Bangla words
BANGLA_PATTERN = re.compile(r'[ঀ-৿]+')

with open(filename+".txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

    # Remove English words from other lines and keep only Bangla words
    res = ""
    for line in lines:
        if "Sample:" in line:
            res+=line+'\n'

        if "Sample:" not in line:
            filtered_line = re.findall(BANGLA_PATTERN, line)
            filtered_line =" ".join(filtered_line)
            if not filtered_line.isspace():
                res+=filtered_line+'\n'

file_v1 = filename+"_v1.txt"
with open(file_v1, 'w') as file:
    file.write(res)
print("FINISHED V1")

#remove remove any empty line from the file 
subprocess.run(['sed', '-i', '/^$/d', file_v1])

with open(file_v1, 'r') as file:
    lines = file.readlines()

cl_line = ''
empty_list_flag = False

for line in lines:
    if line.startswith('Sample'):
        if empty_list_flag: #When we encounter a new "Sample" line, if the previous line was also a "Sample" line, we append an empty list
            cl_line += str([]) + '\n'
        empty_list_flag = True
    else:
        empty_list_flag = False
        line = line.strip()
        line = line.split()
        cl_line += str(line)+'\n'

file_v2 = filename+"_v2.txt"
with open(file_v2, 'w') as file:
    file.write(cl_line)
print("FINISHED V2")

