# ===== < MODULE INFO > =====
# Author: Waddles
# Version: a0
# Description: A smarter logging platform combining live output and logfile functions.
# ===== < IMPORTS & CONSTANTS > =====
import inspect, os, functools, sys
from time import strftime, sleep

FORMATS = {"RED":"\033[31m",
          "YELLOW":"\033[33m",
          "GREEN":"\033[32m",
          "BLUE":"\033[36m",
          "MAGENTA":"\033[35m",
          "GRAY":"\033[90m",
          "CLEAR":"\033[0m"}
VFLAGS = {0:f"<{FORMATS['RED']}FAIL{FORMATS['CLEAR']}>",
          1:f"<{FORMATS['YELLOW']}WARN{FORMATS['CLEAR']}>",
          2:f"<{FORMATS['GREEN']}GOOD{FORMATS['CLEAR']}>",
          3:f"<{FORMATS['BLUE']}INFO{FORMATS['CLEAR']}>",
          4:f"<{FORMATS['MAGENTA']}DBUG{FORMATS['CLEAR']}>"}
MVFLAGS = {0: "<FAIL>",
           1: "<WARN>",
           2: "<GOOD>",
           3: "<INFO>",
           4: "<DBUG>"}

# Printer vars
vtalk = 2 # Highest verbosity level to print at
mono = False # Whether to print using color (written logs never use color)

# Logger vars
vwrite = 3 # Highest verbosity level to write at
logqueue = []
logpath = "logs/" # Where to write logs to
isNew = True # Indicates whether to print linebreak for new execution time
# ===== < BODY > =====
def log(verbosity: int, message: int):
    """
    Log an event with a message and a verbosity (severity) level.

    Parameters
    ----------
    ``verbosity`` : int
        Verbosity level for this message (0=fail,1=warn,2=good,3=info,4=dbug)
    ``message`` : str
        Message to print
    """
    # Step 0: Stack trace for log header
    caller_function = str(inspect.stack()[1].function) # This will retrieve the function from which this was called
    if caller_function == "<module>": caller_function = "__main__"
    caller_location = str(inspect.stack()[1].filename.split("\\")[-1]) # This will retrieve the module that contains that function

    # Step 1: Format and build logstrings
    for punc in [".", ",", "!", "?"]: message = message.rstrip(punc)
    full_color = f"{VFLAGS[verbosity]}{FORMATS['GRAY']}:{caller_location}:{caller_function}()->{FORMATS['CLEAR']}{message}"
    full_monochrome = f"{MVFLAGS[verbosity]}:{caller_location}:{caller_function}()->{message}"

    # Step 2: Tell and write
    if mono: _logPrint(verbosity, full_monochrome)
    else: _logPrint(verbosity, full_color)

    if verbosity <= vwrite: logqueue.append(f"|{strftime('%d-%m-%y %H:%M:%S')}| {full_monochrome}")
    return True

# ===== < DECORATORS > =====
def sapiDumpOnExit(func):
    """
    DECORATOR
    ---------
    Dump the log transaction queue at this function's termination.
    """
    @functools.wraps(func)
    def logDump(*args, **kwargs):
        """
        Dump the log transaction to the file.

        This should only occur when the transaction reaches a set size, or the program terminates.
        """
        log(4, "Decorator function invoked")

        retval = func(*args, **kwargs) # <-- call the function

        # Step 0: Filepath generation & verification
        global logqueue, isNew
        fpath = f"{logpath}{strftime('%d-%m-%y')}.slog"
        if not os.path.exists(fpath):
            with open(fpath, "a+") as _: pass
            log(2, "*.slog file missing, created a new one")
        if os.path.getsize(fpath) > 1000000: log(2, "Slog file exceeding 1MB, recommend regeneration")

        # Step 1: Final log & dump queue
        log(2, f"Wrote {len(logqueue)+1} lines to {fpath}")

        with open(fpath, "a+") as fo:
            if isNew:
                fo.write("---\n")
                isNew = False
            fo.writelines([x+"\n" for x in logqueue])
            logqueue = []
        return retval
    return logDump

def sapiDebug(func):
    """
    DECORATOR
    ---------
    Pre- and post- function, report all stack information in a debug log.
    """
    @functools.wraps(func)
    def prepost(*args, **kwargs):
        log(4, "Decorator function invoked")
        # Pre-function collection
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]

        retval = func(*args, **kwargs) # Function call & result collection

        log(4, f"Invoking <{func.__name__}({', '.join(args_repr + kwargs_repr)})> yielded <{retval!r}>") # Call to sapilog

        return retval # Not eating results
    return prepost

def sapiSlowdown(func):
    """
    DECORATOR
    ---------
    Slows down execution by waiting half a second before invocation.
    """
    @functools.wraps(func)
    def slowdown(*args, **kwargs):
        log(4, "Decorator function invoked")
        # Sleep call
        sleep(.5)
        return func(*args, **kwargs)
    return slowdown

# ===== < HELPERS > =====
def _logPrint(v, m):
    """
    Write an event to the screen.
    """
    if v <= vtalk: print(m)
    return True

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

# ===== < EXCEPTIONS > =====
class SapiException(Exception):
    """
    Base exception for `sapilog` exceptions. All exception classes derived from this will dump the log queue when raised.
    """
    @sapiDumpOnExit
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
