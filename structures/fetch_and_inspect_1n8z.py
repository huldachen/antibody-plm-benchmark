#!/usr/bin/env python3
"""
E4 groundwork — fetch and inspect the trastuzumab-HER2 crystal complex (1N8Z).

1N8Z is the trastuzumab Fab bound to the HER2 extracellular domain — the crystal
complex the E4 structure arm grafts variant CDR-H3s onto (spec §5). This cheap,
no-GPU step downloads the PDB and identifies the antibody heavy/light and HER2
chains and the CDR-H3 loop residues (the Absci WT HCDR3 = SRWGGDGFYAMDY), so the
grafting step later knows exactly which residues to replace.

Run:
    python structures/fetch_and_inspect_1n8z.py
Outputs:
    structures/pdb/1n8z.pdb (cached), stdout chain/CDR-H3 report.
"""

import os
import sys
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDB_DIR = os.path.join(PROJECT_ROOT, "structures", "pdb")
PDB_PATH = os.path.join(PDB_DIR, "1n8z.pdb")
PDB_URL = "https://files.rcsb.org/download/1N8Z.pdb"
WT_HCDR3 = "SRWGGDGFYAMDY"        # Absci parent (trastuzumab) CDR-H3

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def fetch() -> None:
    os.makedirs(PDB_DIR, exist_ok=True)
    if os.path.exists(PDB_PATH):
        print(f"  [cached] {os.path.relpath(PDB_PATH, PROJECT_ROOT)}")
        return
    print(f"  downloading {PDB_URL} ...")
    urllib.request.urlretrieve(PDB_URL, PDB_PATH)
    print(f"  saved {os.path.relpath(PDB_PATH, PROJECT_ROOT)}")


def main() -> None:
    fetch()
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    parser = PDBParser(QUIET=True)
    model = parser.get_structure("1n8z", PDB_PATH)[0]

    print(f"\n1N8Z chains (model 0):")
    chain_seqs = {}
    for chain in model:
        residues = [r for r in chain if is_aa(r, standard=True)]
        seq = "".join(THREE_TO_ONE.get(r.resname, "X") for r in residues)
        chain_seqs[chain.id] = (residues, seq)
        print(f"  chain {chain.id}: {len(residues)} aa residues")

    print(f"\nLocating CDR-H3 (WT HCDR3 = {WT_HCDR3}):")
    for cid, (residues, seq) in chain_seqs.items():
        pos = seq.find(WT_HCDR3)
        if pos >= 0:
            loop = residues[pos:pos + len(WT_HCDR3)]
            first, last = loop[0], loop[-1]
            print(f"  -> HEAVY chain = '{cid}'. CDR-H3 at seq index {pos}, "
                  f"PDB residues {first.id[1]}-{last.id[1]} "
                  f"({first.resname}{first.id[1]} .. {last.resname}{last.id[1]})")
    # Heuristic chain roles.
    print(f"\nChain-role heuristic (by length):")
    for cid, (residues, seq) in sorted(chain_seqs.items(),
                                       key=lambda kv: -len(kv[1][0])):
        role = ("HER2 antigen (long)" if len(residues) > 300
                else "antibody Fab chain (heavy/light ~210-220)")
        print(f"  chain {cid}: {len(residues)} aa -> {role}")


if __name__ == "__main__":
    main()
