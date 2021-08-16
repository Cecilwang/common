import pickle


def save(obj, name):
    with open(name, "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(name, "rb") as f:
        return pickle.load(f)
