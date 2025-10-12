from dataclasses import dataclass
import numpy as np


# TODO: Should this be used as an ABS for classes Annuitants, LifeAssurancePolicyHolders, etc?
# TODO: Should I create one instance per policyholder, or a single instance for all?
@dataclass
class PolicyHolders:
    # TODO: Should this take a single dataframe with all required inputs, rather than multiple np.ndarrays?
    ages: np.ndarray
    select: np.ndarray  # TODO: Can I indicate that this should be a numpy array of booleans? / Should this be an int indicating number of select years remaining?
    sex: np.ndarray