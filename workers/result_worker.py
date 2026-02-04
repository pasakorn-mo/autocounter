import multiprocessing as mp

class ResultWorker(mp.Process):
    def __init__(self, q):
        super().__init__()
        self.q = q

    def run(self):
        while True:
            item = self.q.get()
            print("DB EVENT:", item)
