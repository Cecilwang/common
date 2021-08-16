class cprint(object):

    def __init__(self, color=None):
        self.color = {"red": "\033[91m"}.get(color, None)
        self.end = "\033[0m"

    def __call__(self, str):
        if self.color:
            print(self.color + str + self.end)
        else:
            print(str)
