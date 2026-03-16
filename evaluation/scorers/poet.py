"""PoET model scorer stub.

Implement the `score` function to compute conditional log-likelihoods
using the PoET model.
"""
from pathlib import Path
from Bio import SeqIO


def score(conditioning_fasta, candidates_fasta):
    """Score candidate sequences conditioned on the conditioning set.

    Args:
        conditioning_fasta: Path to conditioning set FASTA
        candidates_fasta: Path to candidate sequences FASTA

    Returns:
        dict of {seq_id: float} where float is the conditional log-likelihood
    """
    raise NotImplementedError(
        "PoET scorer not yet implemented. "
        "Implement this function to call the PoET API or local model."
    )
