from DSSPparser import parseDSSP


def parse_dssp(dssp_path):
    dssp = parseDSSP(dssp_path)
    try:
        dssp.parse()
        return dssp
    except Exception:
        return None
