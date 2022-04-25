
numbered_map = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
                19: "nineteen", 20: "twenty", 30: "thirty", 40: "fourty", 50: "fifty", 60: "sixty", 70: "seventy", 80: "eighty", 90: "ninty", 100: "hundred", 1000: "onethousand"}


def generateTextNumber(number):
    if number in numbered_map:
        if number == 100:
            return "one" + numbered_map[number]
        return numbered_map[number]

    if number >= 100:
        if number % 100 == 0:
            return numbered_map[number // 100] + numbered_map[100]
        return numbered_map[number // 100] + numbered_map[100] + "and" + generateTextNumber(number % 100)

    if number > 10:
        return numbered_map[(number // 10)*10] + numbered_map[number % 10]


summation = 0
for i in range(1, 1001):
    summation += len(generateTextNumber(i))


print(summation)
