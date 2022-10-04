# ===== < INFO > =====
# Author      : William "Waddles" Waddell
# Date        : 3 Sep 2022
# Description : Generates graph-based dataset for training.
# Version     : a1.0
# ===== < IMPORTS & CONSTANTS > =====
from random import uniform, randint
from helpers import sapilog as sl

OUTDIR = "data/" # Where to write the data to

# ===== < BODY > =====
def generateData(degree: int = 3, cfrange: tuple[int] = (-20, 20), srange: tuple[int] = (-100, 100), n_samples: int = 1000, name: str = None) -> dict:
    """
    Generate a dataset for training based on a randomly-generated polynomial graph in R^2 space.

    Parameters
    ----------
    `degree` : int, default = 2
        Degree of the polynomial
    `cfrange` : tuple[int], default = (-20, 20)
        Possible range for the coefficients (result will be a double rounded to 3 decimals)
    `srange` : tuple[int], default = (-100, 100)
        Domain AND range boundaries over which to collect samples
    `n_samples` : int, default = 100
        Number of samples to select
    `name` : str, optional
        Name of the output file; if absent, will not save generated points

    Returns
    -------
    dict[tuple[int]] : int
        Dictionary where keys are (x, y) pairs, and values are either 0 or 1 depending on if y is below or above graph, respectively
    """
    # Step 1: Create graph generating lambda
    components = []
    for d in reversed(range(degree)):
        coefficient = round(uniform(cfrange[0], cfrange[1]), 3)
        components.append(lambda x, d=d, coefficient=coefficient : coefficient * (x ** d)) # Scope loop vars as default parameters
        sl.log(4, f"Component created | {coefficient}x^{d}")
    fullgen = lambda x : sum([f(x) for f in components])

    # Step 2: Collect samples as (x,y):a
    samples = {}
    for i in range(n_samples):
        x, y = randint(srange[0], srange[1]), randint(srange[0], srange[1])

        if (x, y) in samples.keys(): continue # Check if this value has been selected, and stop processing if so (saves time when n_samples is large and srange is small)

        a = fullgen(x)
        # If point falls below line, expected = 0, else expected = 1
        if y < a: a = 0
        else: a = 1
        samples[(x, y)] = a

    # Step 3: Save data, if necessary
    if name:
        sl.log(3, f"Writing {len(samples)} samples to {OUTDIR}{name}.data")
        with open(f"{OUTDIR}{name}.data", "w") as fo:
            for k, v in zip(samples.keys(), samples.values()):
                fo.write(f"{k} = {v}\n")

    return samples

def collectData(source: str, folds: int = 3) -> list[dict]:
    """
    Read data from a .data file, split it into folds, and return it as dictionaries.

    Parameters
    ----------
    `source` : str
        Name of the .data file, without extension. Only include folder structure if not specified in `OUTDIR`
    `folds` : int, default = 3
        Number of folds to split data into

    Returns
    -------
    list[dict]
        List of dictionaries of data-label pairs

    Notes
    -----
    The final datapoint will always be missing due to round-down; the penultimate datapoint may be missing depending on parity of dataset length
    """
    # Step 1: Verify filepath and read raw data
    with open(f"{OUTDIR}{source}.data", "r") as fo:
        data = fo.readlines()
    sl.log(3, f"Read {len(data)} lines from {OUTDIR}{source}.data")

    # Step 2: Format read data back into non-string datatypes
    data = [x.rstrip() for x in data] # Remove trailing newlines and whitespace
    data = [x.split(" = ") for x in data] # Split into tuple of ((x,y), a)
    data = [(eval(x[0]), int(x[1])) for x in data] # Convert key, value into tuple[int] : int

    # Step 3: Fold data, convert to dict, and return
    folded = []
    increment = int(len(data)/folds)

    for f in range(folds):
        temp = data[f*increment:(f+1)*increment]
        fdict = {}
        for k, v in temp:
            fdict[k] = v
        folded.append(fdict)

    sl.log(4, f"Folded {len(data)} points into {folds} {increment}-long sets")
    return tuple(folded)

# ===== < MAIN > =====
if __name__ == "__main__":
    pass
