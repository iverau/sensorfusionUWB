from math import factorial


def combinations(n, r):
    return factorial(n)/(factorial(r)*factorial(n-r))


def determineFactorialsAboveAMillion(index):
    summation = 0
    for i in range(1, index):
        if combinations(index, i) > 1000000:
            summation += 1
    return summation


summation = 0
for i in range(1, 101):
    print(i)
    summation += determineFactorialsAboveAMillion(i)

print(summation)

print(28433*pow(2, 7830457, 10000000000) + 1)
