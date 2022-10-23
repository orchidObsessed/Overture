# ===== < MODULE INFO > =====
# Author: William "Waddles" Waddell
# Version: 2.1.4
# Description: A small threading library using pooling to allow for faster execution of repetitive functions.
# ===== < IMPORTS & CONSTANTS > =====
import threading as t
from queue import Queue, Empty
import functools, inspect

UNIQUE_POOLS = {}
# ===== < BODY > =====
class OldLagoon:
    """

    """
    NAMECOUNTER = 0
    def __init__(self, name="Lagoon", n_procs=1, max_size=0):
        self.name = name + "_" + str(Lagoon.NAMECOUNTER)
        Lagoon.NAMECOUNTER += 1
        self.queue = Queue(maxsize=max_size)
        self.procs = []
        self.spawn_new(n_procs)

    def enqueue(self, fargs_tuple):
        """

        """
        self.queue.put(fargs_tuple)
        return

    def spawn_new(self, n=1):
        """
        Spawn `n` new multiprocessing.Process processes in this pool.
        """
        for i in range(n):
            self.procs.append(t.Thread(target=self.__tick))
            self.procs[-1].start()
        return

    def __tick(self, killcycle=False):
        """

        """
        if not killcycle: end = not t.main_thread().is_alive()
        else: end = self.queue.empty()
        while not end:
            fcall = self.queue.get() # In form: (f, *args) || (!break, n)
            if fcall[0] == "!break":
                self.queue.task_done()
                if fcall[1] > 1: self.queue.put(("!break", fcall[1]-1))
                end, killcycle = True, True
            else:
                try: fcall[0](*fcall[1])
                except NameError as e: sl.log(0, f"NameError in queued function {fcall[0]} with provided arguments {fcall[1]} do not exist; {e}", inspect.stack())
                except TypeError as e: sl.log(0, f"TypeError in queued function {fcall[0]} with given arguments {fcall[1]}; {e}", inspect.stack())
                except Exception as e: sl.log(0, f"Unknown error occured with queued function {fcall[0]} and arguments {fcall[1]}; {e}", inspect.stack())
                finally: self.queue.task_done()
            if not killcycle: end = not t.main_thread().is_alive()
            else: end = self.queue.empty()
        if not killcycle: self.__tick(killcycle=True)
        return

    def join_and_kill(self, n=None):
        """
        Kill `n` processes. If `n` is not present, kills all processes.
        """
        if n: self.queue.put(("!break", n))
        else: self.queue.put(("!break", len(self.procs)))
        self.queue.join()
        return

class OvercomplicatedLagoon:
    """

    """
    NAME_COUNTER = 0

    def __init__(self, n_threads=1, block_time=1):
        self.name = f"Lagoon_{Lagoon.NAME_COUNTER}"
        print(f"Starting construction of Lagoon {self.name}")
        Lagoon.NAME_COUNTER += 1

        self.queue = Queue()
        self.block_time = block_time

        self.size = n_threads # Number of threads at start
        self.__procs = []
        self.spawn(self.size)
        print(f"Spawn finished!")
        self.spin_up()
        print(f"Lagoon with name {self.name} created")

    def enqueue(self, fargs):
        """
        Add a new item to the queue for this pool.

        Parameters
        ----------
        `fargs` : tuple
            Tuple containing callable function at index 0

        Notes
        -----
        This is currently unsafe due to lack of support for pure-args, pure-kwargs, and mixed argtypes.
        """
        print(f"Lagoon with name {self.name} is putting {fargs} in the queue")
        try: self.queue.put(fargs)
        except Exception as e: print(f"Something went wrong when enqueueing item | {e}")
        finally: return

    def spawn(self, n: int = 1):
        """
        Add a new thread to this pool to increase performance for the given function.

        Parameters
        ----------
        `n` : int, default=1
            Number of new threads to add
        """
        print(f"Lagoon with name {self.name} trying to spawn {n} threads...")
        for i in range(n):
            self.__procs.append(t.Thread(target=self.__main()))
        print(f"Lagoon with name {self.name} spawned {n} threads")
        return

    def spin_up(self):
        """

        """
        for proc in self.__procs:
            try:
                proc.start()
            except Exception as e:
                print(f"Process was likely already running | {e}")

        print(f"Lagoon with name {self.name} started all threads")
        return

    def spin_down(self):
        """

        """
        for i in range(self.size): self.enqueue("!")
        self.queue.join() # Block until queue is empty
        return

    def despawn(self, n: int = 0):
        """
        Begin graceful termination and destruction of threads in this pool.

        Parameters
        ----------
        `n` : int, optional
            Number of threads to terminate. If 0, all threads are terminated.
        """
        for i in range(n):
            self.enqueue("!")
        return

    def __main(self):
        """
        Thread's main (identity) loop.
        """
        print(f"Lagoon with name {self.name} begun main loop")
        while True:
            state = self.__work_tick() # Do a work tick and get its return code
            if state == -1:
                print(f"Thread {self.name} ended its work tick with code -1; aborting")
                sys.exit()
            elif state == 2:
                print(f"Thread {self.name} ended its work tick with code 2; starting cleanup")
                return
            elif state == 3:
                print(f"Thread {self.name} recieved termination signal; starting cleanup")
                return
            print(f"Lagoon with name {self.name} finished main loop")
        return

    def __work_tick(self):
        """
        Work tick.

        Attempts to get item from queue and complete it, then returns to the main loop.

        Returns
        -------
        -1 : Something went horribly wrong
        0  : Function was retrieved and executed successfully
        1  : No function was retrieved before timeout
        2  : Function was retrieved and failed to execute
        3  : Graceful termination signal recieved
        """
        end_state = -1 # Default state; this should never be returned
        try:
            fcall = self.queue.get(block=True, timeout=self.block_time) # Get item from queue
            if fcall[0] == "!": end_state = 3
            else:
                fcall[0](*fcall[1])
                end_state = 0
        except Empty as _:
            end_state = 1
        except Exception as e:
            print(f"{self.name} encountered error {e}")
            end_state = 2
        finally:
            if not self.queue.empty(): self.queue.task_done()
        print(f"Lagoon with name {self.name} finished work tick with code {end_state}")
        return end_state

class SimpleLagoon:
    """
    Pooled, uni-threaded task queue.
    """
    def __init__(self, fname: str):
        self.name = str(fname)
        self.__task_queue = Queue()
        self.__proc = t.Thread(target=self.__main).start()
        self.__block_queue = False

    def __main(self):
        main_is_alive = t.main_thread().is_alive()
        queue_is_full = self.__task_queue.full()
        while main_is_alive or queue_is_full:
            try:
                fcall = self.__task_queue.get(block=True, timeout=1) # Function call
                if fcall[0] == "!":
                    break
                else:
                    fcall[0](*fcall[1])
                    self.__task_queue.task_done()
            except Empty as _: pass
            except Exception as e: print(f"Error in Lagoon {self.name}: {e}")
            finally:
                main_is_alive = t.main_thread().is_alive()
                queue_is_full = self.__task_queue.full()
        return

    def enqueue(self, fargs):
        if not self.__block_queue: self.__task_queue.put(fargs)
        else:
            print(f"{fargs} is not being queued because this lagoon is slated for termination!")
        return

def pooled_threaded(func):
    """
    Carry out the decorated function in a thread; do not wait if no threads are available (raises `Exception`).

    This does not return the output of the function.
    """
    global UNIQUE_POOLS
    if func not in UNIQUE_POOLS.keys():
        UNIQUE_POOLS[func] = SimpleLagoon(func)
    thispool = UNIQUE_POOLS[func]
    @functools.wraps(func)
    def wrapper_doInThread(*args):
        thispool.enqueue((func, args))
    return wrapper_doInThread

# ===== < MAIN > =====
if __name__ == "__main__":
    print(f"This shouldn't have been run!")
