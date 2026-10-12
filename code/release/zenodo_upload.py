"""Upload the packaged data to the Zenodo draft and set its metadata. Never publishes.

Token: read from ~/.zenodo_token (scope deposit:write); never printed or logged.
Usage:
  python zenodo_upload.py --check        read-only: show draft state and files
  python zenodo_upload.py --metadata     update metadata only
  python zenodo_upload.py --upload       upload missing/changed files, verify md5, then update metadata
Re-runnable: files already on Zenodo with the same md5 are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

from release_config import STAGING_DIR, ZENODO_RECORD_ID

API = f"https://zenodo.org/api/deposit/depositions/{ZENODO_RECORD_ID}"
UPLOAD = STAGING_DIR / "_upload"
LOG = UPLOAD / "zenodo_upload_log.json"
ATTEMPTS = 3

CREATORS = [
    {"name": "Valantinas, Adomas", "affiliation": "University of Cambridge; Brown University; ETH Zurich"},
    {"name": "Mustard, John F.", "affiliation": "Brown University"},
    {"name": "Tullo, Adriano", "affiliation": "INAF, Osservatorio Astronomico di Padova"},
    {"name": "Bramble, Michael S.", "affiliation": "Jet Propulsion Laboratory, California Institute of Technology"},
    {"name": "Head, James W.", "affiliation": "Brown University"},
]
DESCRIPTION = """<p>Derived data products for Valantinas, A., Mustard, J. F., Tullo, A., Bramble, M. S., &amp; Head, J. W. (2026),
<em>An exposed magmatic feeder system for the regional olivine unit in Nili Fossae, Mars</em>,
Earth and Planetary Science Letters, 696, 120390, <a href="https://doi.org/10.1016/j.epsl.2026.120390">doi:10.1016/j.epsl.2026.120390</a>.</p>
<p>Study site: the "Horseshoe" site in western Nili Fossae, Mars. Contents (one zip per section; unzip all into one folder):</p>
<ul>
<li><b>A_geologic_map</b>: hand-drawn geologic map (GeoPackage, File Geodatabase, shapefiles).</li>
<li><b>B_ArcGIS_project</b>: ArcGIS Pro project used for the paper figures, with a table of every layer source.</li>
<li><b>C_Brown2020_olivine_1um_analysis</b>: 1 &micro;m olivine band fits (Brown et al., 2020 method) for CRISM FRT00003E12, FRT0000871C, FRT0000A4FC, FRT0000B438.</li>
<li><b>D_CRISM_band_parameters</b>: Viviano-Beck et al. (2014) parameter maps, PCA/GMM maps and feature stacks.</li>
<li><b>E_contact_metamorphism</b>: outputs of the contact-metamorphism and band-centre tests, with their ratioed-cube inputs.</li>
<li><b>F_pansharpened_CaSSIS_HiRISE</b>: 26 CaSSIS colour scenes pansharpened with HiRISE (0.25-0.29 m; uint16, I/F = DN &times; 2e-5).</li>
<li><b>G_HiRISE_DEMs_Tullo</b>: two HiRISE stereo DEMs and orthoimages (A. Tullo).</li>
<li><b>H_CTX_DEM</b>: CTX stereo DEM (cleaned and interpolated).</li>
</ul>
<p>Each section contains a README.txt and a MANIFEST.sha256. Raw CRISM, HiRISE, CTX and CaSSIS data are not redistributed;
their product IDs and the analysis code are at <a href="https://github.com/euchas/nili-fossae-crism">github.com/euchas/nili-fossae-crism</a>.</p>"""
METADATA = {
    "upload_type": "dataset",
    "title": "Data for: An exposed magmatic feeder system for the regional olivine unit in Nili Fossae, Mars",
    "creators": CREATORS,
    "description": DESCRIPTION,
    "access_right": "open",
    "license": "cc-by-4.0",
    "keywords": ["Mars", "Nili Fossae", "olivine", "intrusions", "CRISM", "CaSSIS", "HiRISE",
                 "pansharpening", "geologic map", "DEM", "contact metamorphism"],
    "related_identifiers": [
        {"identifier": "https://github.com/euchas/nili-fossae-crism", "relation": "isSupplementedBy",
         "resource_type": "software", "scheme": "url"},
        {"identifier": "10.1016/j.epsl.2026.120390", "relation": "isSupplementTo", "scheme": "doi"},
    ],
    "grants": [{"id": "10.13039/501100001711::217848"}],
}


def session() -> requests.Session:
    tok = (Path.home() / ".zenodo_token").read_text().strip()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {tok}"
    return s


def md5(path: Path, bufsize: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


def get_draft(s: requests.Session) -> dict:
    r = s.get(API, timeout=120)
    r.raise_for_status()
    d = r.json()
    if d.get("submitted"):
        sys.exit("Record is already published; refusing to modify it.")
    return d


def remote_files(s: requests.Session) -> dict[str, dict]:
    r = s.get(f"{API}/files", timeout=120)
    r.raise_for_status()
    return {f["filename"]: f for f in r.json()}


def update_metadata(s: requests.Session) -> None:
    draft = get_draft(s)
    meta = dict(METADATA)
    meta["prereserve_doi"] = True
    if draft["metadata"].get("publication_date"):
        meta["publication_date"] = draft["metadata"]["publication_date"]
    r = s.put(API, json={"metadata": meta}, timeout=120)
    if r.status_code == 400 and "grants" in r.text:
        print("grant id rejected by Zenodo; retrying without it (add funding by hand)")
        meta.pop("grants")
        r = s.put(API, json={"metadata": meta}, timeout=120)
    if not r.ok:
        sys.exit(f"metadata update failed: HTTP {r.status_code} {r.text[:500]}")
    print("metadata updated; DOI", r.json()["metadata"].get("prereserve_doi", {}).get("doi"))


def upload(s: requests.Session) -> None:
    draft = get_draft(s)
    bucket = draft["links"]["bucket"]
    local = json.loads((UPLOAD / "upload_list.json").read_text())
    remote = remote_files(s)
    log = json.loads(LOG.read_text()) if LOG.exists() else {}
    wanted = {e["file"] for e in local}
    placeholder = remote.get("README.txt")
    if placeholder and placeholder["filesize"] < 1000:
        s.delete(f"{API}/files/{placeholder['id']}", timeout=120).raise_for_status()
        remote.pop("README.txt")
        print("removed placeholder README.txt")
    for name, f in remote.items():
        if name not in wanted:
            print(f"warning: remote file not in upload list: {name}")
    for e in local:
        path = UPLOAD / e["file"]
        digest = log.get(e["file"], {}).get("md5") or md5(path)
        if e["file"] in remote and remote[e["file"]]["checksum"] == digest:
            print(f"skip {e['file']} (already uploaded, md5 matches)")
            continue
        r = None
        for attempt in range(1, ATTEMPTS + 1):
            print(f"upload {e['file']} ({path.stat().st_size / 1e9:.2f} GB), attempt {attempt} ...", flush=True)
            try:
                with open(path, "rb") as fh:
                    r = s.put(f"{bucket}/{e['file']}", data=fh, timeout=(60, 7200),
                              headers={"Content-Type": "application/octet-stream"})
                if r.ok:
                    break
                print(f"  HTTP {r.status_code}: {r.text[:200]}", flush=True)
            except requests.exceptions.RequestException as exc:
                print(f"  connection error: {exc}", flush=True)
            time.sleep(60 * attempt)
        if r is None or not r.ok:
            sys.exit(f"upload failed for {e['file']} after {ATTEMPTS} attempts")
        remote_md5 = r.json().get("checksum", "").removeprefix("md5:")
        if remote_md5 != digest:
            sys.exit(f"md5 mismatch for {e['file']}: local {digest}, remote {remote_md5}")
        log[e["file"]] = {"md5": digest, "bytes": path.stat().st_size, "verified": True}
        LOG.write_text(json.dumps(log, indent=2))
        print("  ok, md5 verified", flush=True)


def check(s: requests.Session) -> None:
    d = get_draft(s)
    print("state:", d["state"], "| DOI:", d["metadata"].get("prereserve_doi", {}).get("doi"))
    for name, f in sorted(remote_files(s).items()):
        print(f"  {name}  {f['filesize'] / 1e9:.3f} GB  {f['checksum']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--metadata", action="store_true")
    g.add_argument("--upload", action="store_true")
    a = ap.parse_args()
    s = session()
    if a.check:
        check(s)
    elif a.metadata:
        update_metadata(s)
    else:
        upload(s)
        update_metadata(s)
        check(s)


if __name__ == "__main__":
    main()
