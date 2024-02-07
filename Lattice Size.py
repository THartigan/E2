import math


def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]

    return c
def mag(x): 
    return math.sqrt(sum(i**2 for i in x))
def sqrmag(x): 
    return (sum(i**2 for i in x))
def findR0(A, B):
    print(math.sqrt((3 * (A ** 2)) / (B - math.sqrt((B ** 2) - (9 * (A ** 2))))))
    #print(math.sqrt((3 * (A ** 2)) / (B + math.sqrt((B ** 2) - (9 * (A ** 2))))))

a = [1, 0, 0]
b = [1, 1, 0]
c = [0, 1, 0]

d = 0
while d == 0:
    C = input("input vector 'a' in form 'x,y'")
    C = C.split(",")
    print(C)
    a[0] = float(C[0])
    a[1] = float(C[1])
    C = input("input vector 'b' in form 'x,y'")
    C = C.split(",")
    b[0] = float(C[0])
    b[1] = float(C[1])
    C = input("input vector 'c' in form 'x,y'")
    C = C.split(",")
    c[0] = float(C[0])
    c[1] = float(C[1])
    A = (2 / math.sqrt(3)) * mag(cross(a, b))
    B = sqrmag(a) + sqrmag(b) + sqrmag(c)
    findR0(A, B)

