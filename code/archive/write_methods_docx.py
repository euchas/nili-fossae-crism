"""Write the EPSL methods Word doc + a 1-page supplementary doc.

Both focus on the Brown 2020 olivine 1 um band fit only. The
unsupervised CRISM clustering work lives in its own
SUMMARY_v9_1.md report and is intentionally excluded from these
methods documents.

Two outputs into outputs/reports/methods_docx/:
  - methods_short.docx           - very brief: one paragraph + one
                                   equation, suitable for the main
                                   methods section
  - methods_supplementary.docx   - ~1-page expansion for the SI

Run: python code/write_methods_docx.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "methods_docx"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def style_doc(doc):
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    pf = style.paragraph_format
    pf.line_spacing = 1.2
    pf.space_after = Pt(6)
    section = doc.sections[0]
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)


def add_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(13 if level == 1 else 11)
    return h


def add_paragraph_justified(doc, text):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def add_equation(doc, text):
    """Centered equation paragraph in italics."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.italic = True
    run.font.name = "Times New Roman"
    run.font.size = Pt(11)
    return p


# =============================================================================
# Short methods - Brown analysis only, one paragraph + one equation
# =============================================================================
def write_short():
    doc = Document()
    style_doc(doc)

    add_heading(doc, "Olivine 1 \u00b5m band fitting", level=1)
    add_paragraph_justified(doc,
        "We fit the four-parameter asymmetric Gaussian profile of Brown "
        "et al. (2020) to the olivine 1 \u00b5m absorption in four CRISM "
        "MTR3 cubes covering the Horseshoe site in western Nili Fossae "
        "(FRT00003E12, FRT0000871C, FRT0000A4FC, FRT0000B438). For each "
        "pixel the I/F spectrum is continuum-removed across the "
        "0.75\u20131.75 \u00b5m window, transformed to apparent absorbance "
        "A(\u03bd) = \u2212log\u2081\u2080\u00a0\u03b1(\u03bb) in wavenumber "
        "\u03bd = 10\u2074/\u03bb (cm\u207b\u00b9), and fit by Nelder\u2013Mead "
        "simplex (random seed 42, no parameter bounds) to"
    )
    add_equation(doc,
        "A(\u03bd) = \u03b1 \u00b7 exp[ \u2212(\u03bd \u2212 \u03bd\u2080)\u00b2 / "
        "(\u03c3 \u00b7 (1 + sgn(\u03bd \u2212 \u03bd\u2080) \u00b7 \u03c7))\u00b2 ],"
    )
    add_paragraph_justified(doc,
        "yielding pixelwise centroid \u03bb\u2080 = 10\u2074/\u03bd\u2080, amplitude "
        "\u03b1, half-width \u03c3, and asymmetry \u03c7. The Brown 2020 "
        "Equation 2 quality threshold (1.15 < \u03bb\u2080 < 1.50 \u00b5m, "
        "0.1 < \u03b1 < 1.0, \u03c3 < 0.30, 0.8 < \u03c7 < 9) is recorded "
        "as a separate diagnostic mask but is not used to clip the "
        "underlying parameter arrays, so the surrounding mafic-cap and "
        "phyllosilicate populations enter all downstream products. "
        "Three-band RGB composites (R = centroid, G = amplitude, B = "
        "asymmetry) are produced at four-cube-pooled physical stretches "
        "(centroid 1.15\u20131.40, amplitude 0.10\u20130.37, asymmetry "
        "1.25\u20135.00) so identical color denotes identical parameter "
        "value across cubes. Full implementation details are in the "
        "Supplementary Methods."
    )

    out = OUT_DIR / "methods_short.docx"
    doc.save(out)
    print(f"wrote {out}")


# =============================================================================
# Supplementary - ~1 page expansion of the Brown analysis
# =============================================================================
def write_supplementary():
    doc = Document()
    style_doc(doc)

    add_heading(doc, "Supplementary methods: olivine 1 \u00b5m band fitting",
                level=1)

    add_heading(doc, "S1. CRISM data", level=2)
    add_paragraph_justified(doc,
        "Four CRISM Full Resolution Targeted observations covering the "
        "Horseshoe site in western Nili Fossae were processed in their "
        "map-projected MTR3 form (FRT00003E12, FRT0000871C, FRT0000A4FC, "
        "FRT0000B438). Atmospheric correction was applied during the "
        "standard PDS processing chain (volcano-scan); residual "
        "CO\u2082 contamination near 2.0 \u00b5m is annotated on extracted "
        "spectra. Bad bands flagged in the CRISM header BBL field were "
        "treated as missing values throughout."
    )

    add_heading(doc, "S2. Asymmetric Gaussian fit", level=2)
    add_paragraph_justified(doc,
        "Each pixel spectrum was fit to the asymmetric Gaussian of Brown "
        "et al. (2020) following four steps. (1) Bad CRISM channels in the "
        "0.75\u20131.75 \u00b5m window were interpolated from neighbouring "
        "valid channels. (2) The local continuum was constructed as the "
        "straight line joining reflectance at 0.75 and 1.75 \u00b5m and "
        "the spectrum divided by it to yield continuum-removed reflectance "
        "\u03b1(\u03bb). (3) The spectrum was transformed to apparent "
        "absorbance A(\u03bd) = \u2212log\u2081\u2080\u00a0\u03b1(\u03bb) "
        "and the wavelength axis inverted to wavenumber \u03bd = 10\u2074/\u03bb "
        "(cm\u207b\u00b9), so the absorption appears as a peak. (4) The four-"
        "parameter asymmetric Gaussian was fit to A(\u03bd) by adaptive "
        "Nelder\u2013Mead simplex (scipy.optimize.minimize, tolerance "
        "10\u207b\u2074, max 10\u2074 evaluations, fixed initialisation: "
        "centroid at the absorbance maximum, \u03c3\u2080 = 1500 cm\u207b\u00b9, "
        "\u03c7\u2080 = 1.5). No parameter bounds were imposed during the "
        "fit. Centroids are reported in wavelength as \u03bb\u2080 = "
        "10\u2074/\u03bd\u2080; half-widths are converted approximately as "
        "\u0394\u03bb \u2248 \u03bb\u2080\u00b2 \u00b7 \u03c3 / 10\u2074."
    )

    add_heading(doc, "S3. Quality flagging vs. clipping", level=2)
    add_paragraph_justified(doc,
        "The Brown 2020 Equation 2 reliability threshold "
        "(1.15 < \u03bb\u2080 < 1.50 \u00b5m, 0.1 < \u03b1 < 1.0, \u03c3 < 0.30, "
        "0.8 < \u03c7 < 9) is computed pixelwise and stored as a separate "
        "boolean mask. It is not used to clip the underlying parameter "
        "arrays. This departure from the Brown 2020 protocol preserves "
        "sub-1.15 \u00b5m centroid pixels \u2014 spectrally flat mafic-cap "
        "outcrops and phyllosilicate exposures whose 1 \u00b5m fits are "
        "noisy but whose adjacent VB summary parameters are real \u2014 "
        "for ROI-level inspection. Pixels in the sub-1.15 \u00b5m centroid "
        "tail correspond predominantly to topographic / illumination "
        "geometries on which the asymmetric-Gaussian fit is dominated by "
        "the visible-edge slope, and are flagged as such in the "
        "supplementary maps."
    )

    add_heading(doc, "S4. Map-projected RGB composites", level=2)
    add_paragraph_justified(doc,
        "Per-cube three-band RGB composites are written as 4-band uint8 "
        "RGBA GeoTIFFs at the source MTR3 projection (MRO Mars "
        "Equirectangular, IAU 2000, 18 m / pixel) with R = centroid, "
        "G = amplitude, B = asymmetry. A single matched stretch is "
        "applied across all four cubes \u2014 centroid 1.15\u20131.40 "
        "\u00b5m, amplitude 0.10\u20130.37, asymmetry 1.25\u20135.00 \u2014 "
        "anchored on the Brown 2020 reliability bounds rather than on per-"
        "cube percentiles, so identical color across cubes denotes "
        "identical parameter value. The alpha band carries the finite-fit "
        "mask; pixels outside the cube footprint are transparent. The "
        "four cube composites are merged geographically into a single "
        "mosaic GeoTIFF for figure inclusion."
    )

    add_heading(doc, "S5. Reproducibility", level=2)
    add_paragraph_justified(doc,
        "Every script writes a JSON parameter log with run timestamp, "
        "fixed random seed (42), and source-file MD5 checksums. Pixel-"
        "level Brown fits are deterministic given the seed and the pinned "
        "package environment (numpy, scipy, scikit-learn, rasterio; "
        "versions in requirements.txt). Brown parameter arrays "
        "(.npy), GeoTIFFs, and matched-stretch RGB composites are "
        "archived in outputs/brown_rgb/ alongside the input cubes."
    )

    out = OUT_DIR / "methods_supplementary.docx"
    doc.save(out)
    print(f"wrote {out}")


def main():
    print(f"output dir: {OUT_DIR}")
    write_short()
    write_supplementary()


if __name__ == "__main__":
    main()
