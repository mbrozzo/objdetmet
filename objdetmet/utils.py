def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def cast_if_different_class(obj, class_):
    if isinstance(obj, class_):
        return obj
    return class_(obj)