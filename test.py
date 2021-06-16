def rev(s):
    if len(s) == 0:
        return s
    else:
        return s[-1] + rev(s[0:-1])


x = "abcd"
r = rev(x)
print(r)
