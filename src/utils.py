from DSSPparser import parseDSSP


def parse_dssp_check(dssp_path, seq):
    dssp = parseDSSP(dssp_path)
    try:
        dssp.parse()
        if "".join(dssp.aa) == seq:
            return True
    except Exception:
        return None
