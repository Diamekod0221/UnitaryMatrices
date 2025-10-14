# Pure Python Mersenne Twister
class MT19937:
    def __init__(self, seed: int = 5489):
        # MT19937 parameters
        self.w, self.n, self.m, self.r = 32, 624, 397, 31
        self.a = 0x9908B0DF
        self.u, self.d = 11, 0xFFFFFFFF
        self.s, self.b = 7, 0x9D2C5680
        self.t, self.c = 15, 0xEFC60000
        self.l = 18
        self.f = 1812433253

        # Initialize state array
        self.MT = [0]*self.n
        self.index = self.n
        self.MT[0] = seed
        for i in range(1, self.n):
            self.MT[i] = (self.f * (self.MT[i-1] ^ (self.MT[i-1] >> (self.w-2))) + i) & 0xFFFFFFFF

    def twist(self):
        for i in range(self.n):
            x = (self.MT[i] & 0x80000000) + (self.MT[(i+1) % self.n] & 0x7FFFFFFF)
            xA = x >> 1
            if x % 2 != 0:
                xA ^= self.a
            self.MT[i] = self.MT[(i + self.m) % self.n] ^ xA
        self.index = 0

    def random_uint32(self):
        if self.index >= self.n:
            self.twist()
        y = self.MT[self.index]
        y ^= ((y >> self.u) & self.d)
        y ^= ((y << self.s) & self.b)
        y ^= ((y << self.t) & self.c)
        y ^= (y >> self.l)
        self.index += 1
        return y & 0xFFFFFFFF

    def random(self):
        """Return float in [0,1)"""
        return self.random_uint32() / 0x100000000

# Example usage
mt = MT19937(seed=1234)
for _ in range(5):
    print(mt.random())
