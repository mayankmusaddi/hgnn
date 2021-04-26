lf = open('seqfish_plus/cortex_annotation.txt')
w = open('newfile.csv','w+')

with open('seqfish_plus/cortex_svz_counts.csv') as f:
    for line in f:
        label = lf.readline().strip().split('\t')[2]
        print(label)
        new_line = line.strip()+','+label+'\n'
        w.write(new_line)
w.close()
lf.close()