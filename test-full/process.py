with open('newstest2014-deen-src.en.sgm','r') as f:
    for line in f:
        if '<seg id=' in line:
            line = line.split('>')[1].split('</seg')[0]
            print(line)
