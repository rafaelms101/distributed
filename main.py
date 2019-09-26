from statistics import median, mean, stdev

file = open('/home/rafael/mestrado/distributed/results/mydyn.26914276.comet-33-11.out')
content = file.read()
lines = content.split('\n')
lines = [line for line in lines if len(line) >= 1]

lines = [line.split() for line in lines]


results = {'p': {}, 'c': {}, 'i': {}}
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
        results[dist][alg][load] = {'rt': [], 'iv': []}

    if dist == 'p':
        ls = []
        for i in range(100):
            ls.append(float(next(it)[0]))
        results[dist][alg][load]['iv'].append(ls)

    response_time = float(next(it)[0])
    next(it)

    results[dist][alg][load]['rt'].append(response_time)


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


def gen_rt_table(type, results, op=median):
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
                table += str(op(results[('s', alg)][load]['rt'])) + ' '
            except KeyError:
                table += '* '
        for alg in dyn_algs:
            try:
                table += str(op(results[alg][load]['rt'])) + ' '
            except KeyError:
                table += '* '
        table += '\n'
    return table


# results[dist][alg][load]['iv']
def poisson_compare(results, a1, a2, load):
    def join(ls1, ls2):
        return mean([e1 / e2 for e1, e2 in zip(ls1, ls2)])

    lss1 = results['p'][a1][load]['iv']
    lss2 = results['p'][a2][load]['iv']

    return median([join(ls1, ls2) for ls1, ls2 in zip(lss1, lss2)])


print(gen_rt_table('c', results))
print()
print(gen_rt_table('p', results))

