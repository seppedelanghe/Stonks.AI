def prod(val) : 
    res = 1 
    for ele in val: 
        res *= ele 
    return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unicode_block(s: int):
    assert s > 0 and s < 9, 's to small or too large, needs to be 1-8'
    return [
        u'\u2581',
        u'\u2582',
        u'\u2583',
        u'\u2584',
        u'\u2585',
        u'\u2586',
        u'\u2587',
        u'\u2588',
    ][s-1]