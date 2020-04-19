def load_ref(refname):
    import numpy as np

    data = np.load(refname)

    return(data['x'], data['y'])
