import matplotlib.pyplot as plt
from numpy import average, linspace

from gtpts import gtpts

xs = [pt[0] for pt in gtpts]
ys = [pt[1] for pt in gtpts]

lefts = []
rights = []
for pt in gtpts:
    if pt[0] <= ((max(xs) + min(xs)) / 2):
        lefts.append(pt)
    else:
        rights.append(pt)

for left in lefts:
    plt.plot(left[0], left[1], "ro")
for right in rights:
    plt.plot(right[0], right[1], "bo")

leftsx = [p[0] for p in lefts]
rightsx = [p[0] for p in rights]

leftsy = [p[1] for p in lefts]
rightsy = [p[1] for p in rights]

avgleft = (average(leftsx), average(leftsy))
avgright = (average(rightsx), average(rightsy))

x1, y1, x2, y2 = avgleft[0], avgleft[1], avgright[0], avgright[1]

plt.plot(x1, y1, "go")
plt.plot(x2, y2, "go")


n1 = y2 - y1
n2 = x2 - x1
m = n1 / n2
b = -m * x1 + y1


def f(x):
    return m * x + b


print(f"Found Line! \ny = {m:.2f}x {'+' if b >= 0 else '-'} {abs(b):.2f}")

xline = linspace(min(xs), max(xs), 400)
yline = []
for x in xline:
    yline.append(f(x))

plt.scatter(xline, yline)

mae = 0
for pt in gtpts:
    mae += abs(f(pt[0]) - pt[1])
mae /= len(xs)

print(f"mean absolute error: {mae:.2f}")

# uncomment to show points and line visually
# plt.show()
