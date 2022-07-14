def prod(val) : 
    res = 1 
    for ele in val: 
        res *= ele 
    return res

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)