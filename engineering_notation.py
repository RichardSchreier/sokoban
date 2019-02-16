import math

prefixes_n = ['', 'm', 'u', 'n', 'p', 'f', 'a', 'z', 'y']
prefixes_p = ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']


def eng(x, nsd=3):
    if type(x) is str:  # return a float
        if x[-1] in prefixes_n:
            multiplier = 10 ** (-3 * prefixes_n.index(x[-1]))
            x = x[:-1]
        elif x[-1] in prefixes_p:
            multiplier = 10 ** (3 * prefixes_p.index(x[-1]))
            x = x[:-1]
        else:
            multiplier = 1
        return float(x) * multiplier
    else:   # return a string
        if x >= 0:
            sign = ''
        else:
            sign = '-'
            x = -x
        p = int(math.floor(math.log10(x)))
        x = round(x * 10 ** -p, nsd - 1)
        if x >= 10:
            x /= 10
            p += 1
        p3 = p // 3
        if p3 >= 0:
            p3 = min([p3, len(prefixes_p)])
            prefix = prefixes_p[p3]
        else:
            p3 = -min([-p3, len(prefixes_n)])
            prefix = prefixes_n[-p3]
        dp = p - 3 * p3
        x = x * 10 ** dp
        if nsd > dp:
            fmt = f"%.{nsd - dp - 1}f"
        else:
            fmt = "%g"
        return  sign + (fmt % x) + prefix


if __name__ == '__main__':
    print(eng(1, 4))
    print(eng(3.14159))
    print(eng(-31.4159))
    print(eng(314.159, 4))
    print(eng(314.159, 3))
    print(eng(314.159, 2))
    print(eng(314.159, 1))
    print(eng(1000))
    print(eng(999.9, 4))
    print(eng(999.9, 3))
    print(eng(999.9, 2))
    print(eng(31.416e-6))
    print(eng('31.4k'))
