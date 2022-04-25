

allowed_values = [200, 100, 50, 20, 10, 5, 2, 1]


def amount_of_numbers(values, target):
    if target == 0:
        return 1
    if len(values) == 1:
        return 1
    summation = 0
    for index in range(len(values)):
        if target >= values[index]:
            summation += amount_of_numbers(values[index:],
                                           target - values[index])
    print("Sum", summation)
    return summation


print(amount_of_numbers(allowed_values, 200))
