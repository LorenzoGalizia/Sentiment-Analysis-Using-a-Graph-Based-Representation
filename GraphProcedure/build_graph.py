import itertools
#from itertools import islice
import collections
import igraph as ig

"""window 1-step"""
#def window(seq, n=3):
#    it = iter(seq)
#    result = tuple(islice(it, n))
#    if len(result) == n:
#        yield result
#    for elem in it:
#        result = result[1:] + (elem,)
#        yield result


def window(seq, n=3):
    wind = list()
    for i in range(0, len(seq), n-1):
        wind.append(seq[i:i+n])
    return wind

def co_occ(dataframe):
    df = dataframe['text']
    result = collections.defaultdict(lambda: collections.defaultdict(int))
    for doc in df.index:
        seq = df[doc].split()
        wind = list(window(seq, 3))
        for row in wind:
            counts = collections.Counter(row)
            for key_from, key_to in itertools.combinations(counts, 2):
                if key_to in result[key_from].keys():
                    result[key_from][key_to] += counts[key_to]
                else:
                    result[key_from][key_to] = counts[key_to]
        for word in seq:
            if word not in result.keys():
                result[word] = {}
    l_key = list(result.keys())
    d = dict([(y, x + 1) for x, y in enumerate(sorted(set(l_key)))])
    inv_map = {v - 1: k for k, v in d.items()}
    new_d = {}
    for k, v in result.items():
        one = d[k] - 1
        for key in v.keys():
            two = d[key] - 1
            if (two,one) in new_d.keys():
                pass
            else:
                new_d[(one, two)] = v[key]
    edges, weights = zip(*new_d.items())
    edges = list(edges)
    G = ig.Graph(edges, edge_attrs={"weight": weights}, directed = False)
    G.vs["name"] = list(inv_map.values())
    G.vs["label"] = G.vs["name"]
    G.vs["index"] = list(inv_map.keys())
    G.es["label"] = G.es["weight"]
    return G
