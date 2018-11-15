import random

def select_random(x, number):
    random_features = random.sample(range(len(x[0]) + 1), number)
    subset = []
    for instance in x:
        new_instance = []
        for r in range(len(random_features)):
            new_instance.append(instance[r])
        subset.append(new_instance)
    return subset