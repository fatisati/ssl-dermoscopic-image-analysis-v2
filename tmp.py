class A:
    def __init__(self):
        self.b = 1
        self.c = 2
        self.f = lambda x: print(x)
        self.fname = self.f.__name__

a = A()
print(a.__dict__)
