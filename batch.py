from statistics import median, mean

file = open('/home/rafael/mestrado/distributed/results/bout27.out')
content = file.read()
lines = content.split('\n')
lines = [line for line in lines if len(line) >= 1]

lines = [line.split() for line in lines]

results = {}
it = iter(lines)

for line in it:
    size = int(line[0])
    kind = line[1]
    if size not in results:
        results[size] = {}
    if kind not in results[size]:
        results[size][kind] = []
    next(it)
    tt = float(next(it)[0])
    results[size][kind].append(tt)


sks = set()
for d in results.values():
    for k in d.keys():
        sks.add(k)

tbl = 'Each'
for key in sks:
    tbl += ' ' + key
tbl += '\n'

for size in sorted(results.keys()):
    tbl += str(size)
    for k in sks:
        tbl += ' ' + str(int(50000 / median(results[size][k])))
    tbl += '\n'

print(tbl)
