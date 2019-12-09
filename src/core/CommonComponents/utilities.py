from scoop import futures

USE_SCOOP = True

def multi_repeat(n, funcs):
    if USE_SCOOP:
        fs = [futures.submit(func) for _ in range(n) for func in funcs]
        futures.wait(fs)
        return [f.result() for f in fs]
    else:
        return [func() for _ in range(n) for func in funcs ]

def repeat(func, n):
    return multi_repeat(n, [func])

