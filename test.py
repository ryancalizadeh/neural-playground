import numpy as np

def foo(x: int) -> int:
    return "hello " + x

def tuple_check(tup: tuple):
    print(tup[0])
    a = tup[0]
    a *= 3
    print(tup[0])

def main():
    a = np.ones((4))
    b = np.ones((9))

    tup = (a, b)

    tuple_check(tup)
    print(tup[0])
    print(a)

if __name__=="__main__":
    main()