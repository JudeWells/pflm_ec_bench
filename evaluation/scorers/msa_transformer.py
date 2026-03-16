"""MSA Transformer scorer stub.

Implement the `score` function to compute pseudo-likelihoods
using the MSA Transformer model.
"""
from pathlib import Path
from Bio import SeqIO


def score(conditioning_fasta, candidates_fasta):
    """Score candidate sequences conditioned on the conditioning set.

    Args:
        conditioning_fasta: Path to conditioning set FASTA
        candidates_fasta: Path to candidate sequences FASTA

    Returns:
        dict of {seq_id: float} where float is the pseudo-likelihood
    """
    raise NotImplementedError(
        "MSA Transformer scorer not yet implemented. "
        "Implement this function to use the MSA Transformer model."
    )
