import numpy as np
import random

def categorical(x):
    """
    Samples (symbolically) from categorical distribution using gumbell trick
    wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution

    x: logits
    """
    #z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    #return (x + z).argmax()
    u = np.random.uniform(0,1, len(x))
    return np.argmax(x - np.log(-np.log(u)))

def random_choice(x):
    return random.choices(range(0, len(x)), weights=x)[0]

