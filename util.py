

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    for k, v in fr.items():
        if type(v) is dict and k in to and type(to[k]) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to