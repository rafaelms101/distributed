from statistics import median

file = open('/home/rafael/mestrado/distributed/log')
content = file.read()
lines = content.split('\n')
lines = [line for line in lines if len(line) >= 1]

lines = [line.split() for line in lines]


results = {'p': {}, 'c': {}}
it = iter(lines)

for line in it:
    static = line[0] == 's'
    dist = line[1]
    load = float(line[2])
    alg = line[3]

    if static:
        alg = ('s', int(alg))

    if alg not in results[dist]:
        results[dist][alg] = {}

    if load not in results[dist][alg]:
        results[dist][alg][load] = []

    if dist == 'p':
        for i in range(100):
            next(it)

    response_time = float(next(it)[0])
    next(it)

    results[dist][alg][load].append(response_time)


def list_algs(results):
    algs = set()
    for alg in results:
        algs.add(alg)
    return algs


def list_loads(results):
    loads = set()
    for item in results.values():
        for load in item:
            loads.add(load)
    return loads


def gen_table(type, results):
    results = results[type]
    algs = list_algs(results)
    loads = list_loads(results)
    static_algs = [alg[1] for alg in algs if isinstance(alg, tuple)]
    static_algs.sort()
    dyn_algs = [alg for alg in algs if not isinstance(alg, tuple)]
    dyn_algs.sort()

    table = 'Load '
    for alg in static_algs:
        table += str(alg) + ' '
    for alg in dyn_algs:
        table += alg + ' '
    table += '\n'
    for load in loads:
        table += str(load) + ' '
        for alg in static_algs:
            try:
                table += str(median(results[('s', alg)][load])) + ' '
            except KeyError:
                table += '* '
        for alg in dyn_algs:
            try:
                table += str(median(results[alg][load])) + ' '
            except KeyError:
                table += '* '
        table += '\n'
    return table


print(gen_table('c', results))
print()
print(gen_table('p', results))

