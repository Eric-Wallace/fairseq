import sys
input_file = sys.argv[1]

lines = []
item = []
index = 0
with open(input_file, 'r') as f:
    for line in f:
        if index % 4 == 0 and index > 0:
            lines.append(item)
            item = []
        item.append(line.strip())
        index = index + 1
if item != []:
    lines.append(item)
with open(input_file.split('.raw')[0] + 'sys.out','w') as sysout:
    with open(input_file.split('.raw')[0] + 'ref.out', 'w') as refout:
        with open(input_file.split('.raw')[0] + 'transfer.out', 'w') as transferout:
            for bucket in lines:
                sysout.write(bucket[1] + '\n')
                refout.write(bucket[3] + '\n')
                transferout.write(bucket[0] + '\n')
                transferout.write(bucket[2] + '\n')
