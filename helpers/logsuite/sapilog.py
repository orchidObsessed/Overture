# ===== < MODULE INFO > =====
# Author: William "Waddles" Waddell
# Version: 2.1.4
# Description: A smarter logging platform combining live output and logfile functions.
# ===== < IMPORTS & CONSTANTS > =====
import inspect, os, functools, sys, threading
from time import strftime, sleep
from helpers.logsuite import velocius as vl

# Formatting constants
COLOR = {"RED":"\033[31m",
          "YELLOW":"\033[33m",
          "GREEN":"\033[32m",
          "BLUE":"\033[36m",
          "MAGENTA":"\033[35m",
          "GRAY":"\033[90m",
          "CLEAR":"\033[0m"} # Color codes, for easier referencing
FLAGS = {0:f"<{COLOR['RED']}FAIL{COLOR['CLEAR']}>",
         1:f"<{COLOR['YELLOW']}WARN{COLOR['CLEAR']}>",
         2:f"<{COLOR['GREEN']}GOOD{COLOR['CLEAR']}>",
         3:f"<{COLOR['BLUE']}INFO{COLOR['CLEAR']}>",
         4:f"<{COLOR['MAGENTA']}DBUG{COLOR['CLEAR']}>"} # Prebaked verbosity prefixes
MONO_FLAGS = {0: "<FAIL>",
           1: "<WARN>",
           2: "<GOOD>",
           3: "<INFO>",
           4: "<DBUG>"} # Monochrome verbosity prefixes (for mono-mode logging, or logwrite)

# Log-level constants
MAX_V_PRINT = 2 # Highest verbosity level to print at
MAX_V_WRITE = 3 # Highest verbosity level to write at
MONO = False # Whether to print using color (written logs never use color)

# Logger vars
QUEUE = []
UNIV_LOG_LOCK = threading.Lock()
MAX_QUEUE_LEN = 10 # Max number of lines to save in the queue
MAX_QUEUE_MEM = 32000 # Max number of bytes queue can take up
MAX_LOG_MEM = 1000000 # Max memory size of logfile before warning
LOG_PATH = "helpers\\logsuite\\logs\\" # Where to write logs to

# ===== < BODY > =====
@vl.pooled_threaded
def log(verbosity: int, message: int, stack: list = None):
    """
    Log an event with a message and a verbosity (severity) level.

    Parameters
    ----------
    `verbosity` : int
        Verbosity level for this message (0=fail,1=warn,2=good,3=info,4=dbug)
    `message` : str
        Message to print
    `stack` : list, optional
        Stack trace at time of call.

    Notes
    -----
    Looking for a way to automatically get the stack trace; the introduction of multithreading has made logging much more viable, but harder to trace.
    If `stack` is not present, log will fill in missing and warn.
    """
    # Step 0: If verbosity is too low, skip
    if verbosity > MAX_V_WRITE and verbosity > MAX_V_PRINT: return True

    # Step 1: Stack trace for log header
    if stack:
        caller_function = str(stack[0].function) # This will retrieve the function from which this was called
        if caller_function == "<module>": caller_function = "__main__"
        caller_location = str(stack[0].filename.split("\\")[-1]) # This will retrieve the module that contains that function
    else:
        # log(1, "Stack trace was not given; using filler", inspect.stack())
        caller_location, caller_function = "?.py", "?"

    # Step 2: Format and build logstrings
    for punc in [".", ",", "!", "?"]: message = message.rstrip(punc)
    full_color = f"{FLAGS[verbosity]}{COLOR['GRAY']}:{caller_location}:{caller_function}()->{COLOR['CLEAR']}{message}"
    full_monochrome = f"{MONO_FLAGS[verbosity]}:{caller_location}:{caller_function}()->{message}"

    # Step 3: Tell and enqueue, under with to prevent race condi
    with UNIV_LOG_LOCK:
        global QUEUE
        if verbosity <= MAX_V_PRINT:
            if MONO: print(full_monochrome)
            else: print(full_color)
        else: print(f"Verbosity {verbosity} > max print {MAX_V_PRINT}")
        if verbosity <= MAX_V_WRITE: QUEUE.append(f"|{strftime('%d-%m-%y %H:%M:%S')}| {full_monochrome}")

        if len(QUEUE) >= MAX_QUEUE_LEN or sys.getsizeof(QUEUE) >= MAX_QUEUE_MEM: _log_dump()

    return True

# ===== < HELPERS > =====
def _logBox(m):
    """

    Write a message `m` within a box, using the DBUG flag.
    +--------------------+
    |  Message in a box! |
    +--------------------+
    """
    if len(m) > 20: m = m[:20]
    wspace = int((20-len(m))/2) # This gets floored to preserve hard 20 char limit
    if len(m) % 2 == 0: m = "|" + " "*wspace + m + " "*wspace + "|"
    else: m = "|" + " "*wspace + m + " "*wspace + " |"
    log(4, "\t\t+--------------------+")
    log(4, "\t\t"+m)
    log(4, "\t\t+--------------------+")
    return

def _log_dump():
    global QUEUE

    # Filepath generation & verification
    fpath = f"{LOG_PATH}{strftime('%m-%d-%y')}.slog"
    if not os.path.exists(fpath):
        with open(fpath, "w") as _: pass
    if os.path.getsize(fpath) > 1000000: log(1, "Slog file exceeding 1MB, recommend regeneration", inspect.stack())

    # Dump queue

    with open(fpath, "a+") as fo:
        fo.writelines([x+"\n" for x in QUEUE])
        fo.write("*** "*30 + "\n")
    QUEUE = []
    return

# ===== < EXCEPTIONS > =====
class SapiException(Exception):
    """
    Base exception for `sapilog` exceptions. All exception classes derived from this will dump the log queue when raised.
    """
    def __init__(self):
        pass

# ===== < MAIN > =====
if __name__ == "__main__":
    log(0, "This is a sample failure")
    sleep(0.5)
    log(1, "This is a sample warning")
    sleep(0.5)
    log(2, "This is a sample success message")
    sleep(0.5)
    log(3, "This is a sample info message")
    sleep(0.5)
    log(4, "This is a sample debug message")
    sleep(0.5)
    _logBox("now tabbed for visibility")
