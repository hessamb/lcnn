import sys
import matplotlib.pyplot as plt

def plot(legend):
    data = {}
    for l in legend:
        data[l] = []

    for line in sys.stdin:
        values = line.strip().split()
        assert len(legend) == len(values), "#legend and #values not the same"
        for i in range(len(legend)):
            tries = map(float, values[i].strip('/').split('/'))
            data[ legend[i] ].append( float(sum(tries)) / float(len(tries)) )
    x = range(1, len(data[legend[0]]) + 1)
    for label in legend:
        print("%s -> %s" % (label, data[label]))
        plt.plot(x, data[label], label = label)
    plt.legend(loc='upper left')
    plt.show()

if __name__ == '__main__':
    legend = sys.argv[1:]
    plot(legend)
