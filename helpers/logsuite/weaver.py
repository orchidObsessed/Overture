# ===== < INFO > =====

# ===== < IMPORTS & CONSTANTS > =====
import threading as t
from queue import Queue, Empty
import functools

POOLS = {} # func:SoloPool(func)

# ===== < HELPERS > =====
def pool_wrapper(func):
    """

    """
    global POOLS
    if func not in POOLS.keys(): POOLS[func] = SoloPool(func)
    @functools.wraps(func)
    def inner_pool_wrapper(*args):
        POOLS[func].enqueue(args)
    return inner_pool_wrapper

# ===== < BODY > =====
class SoloPool:
    """"""
    def __init__(self, f: callable, name: str = None):
        self._f = f
        self._q = Queue()
        self._threads = [t.Thread(target=self._tick).start()]
        self.name = name
        if not self.name: self.name = str(self._f.__name__)
        self.n_workers = 0

    def _tick(self):
        """

        """
        main_thread_is_alive = t.main_thread().is_alive()
        queue_is_full = not self._q.empty()
        while main_thread_is_alive or queue_is_full:
            # Try and complete an item from the queue
            self._try_enqueued_call()

            # Update exit conditions
            main_thread_is_alive = t.main_thread().is_alive()
            queue_is_full = not self._q.empty()
        return

    def _try_enqueued_call(self):
        """

        """
        # Step 1: Try for 1 second to get something out of the queue
        try:
            f_args = self._q.get(timeout=.5) # Get it
            self._f(*f_args) # Do it
            self._q.task_done() # Mark it as done
        except Empty: pass # Most likely case: nothing in queue
        # except Exception as e:
        #     print(f"{self.name} encountered error when evaluating function call:\n{str(self._f.__name__)}{f_args}")
        #
        return

    def enqueue(self, item):
        """

        """
        self._q.put(item)
        return
# ===== < MAIN > =====
