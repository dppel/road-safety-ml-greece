"""
Generator for the Greek Presentation & User Guide PDF (`generate_pdf_guide.py`)
===============================================================================
Builds a comprehensive, publication-grade Greek PDF manual explaining:
  - Executive Summary & Data Architecture (ELSTAT SDT04 + Independent Sources)
  - Left Sidebar Navigation & Global KPI Summary
  - Detailed walkthrough of all 7 Dashboard Tabs (Views 1 to 7):
      * Purpose & What the Tab Shows
      * Interactive Controls & Presets
      * KPI Cards & Plain-Language Glossary (Υπομνήματα)
      * Embedded High-Resolution Figures & Tables from the Pipeline
      * Speaker Notes ("Τι να τονίσετε στην παρουσίαση")
  - Quick Q&A Cheat-Sheet for live committee/defense questions.
"""

from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"


def register_greek_fonts() -> tuple[str, str, str]:
    """Registers Windows Segoe UI / Arial Unicode TrueType fonts for Greek text."""
    win_fonts = Path("C:/Windows/Fonts")
    candidates = [
        ("SegoeUI", "segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf"),
        ("ArialGR", "arial.ttf", "arialbd.ttf", "ariali.ttf"),
        ("CalibriGR", "calibri.ttf", "calibrib.ttf", "calibrii.ttf"),
    ]
    for family, reg, bold, ital in candidates:
        if (win_fonts / reg).exists() and (win_fonts / bold).exists():
            pdfmetrics.registerFont(TTFont(family, str(win_fonts / reg)))
            pdfmetrics.registerFont(TTFont(f"{family}-Bold", str(win_fonts / bold)))
            ital_path = win_fonts / ital if (win_fonts / ital).exists() else win_fonts / reg
            pdfmetrics.registerFont(TTFont(f"{family}-Italic", str(ital_path)))
            return family, f"{family}-Bold", f"{family}-Italic"
    return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"


def build_styles(font_reg: str, font_bold: str, font_ital: str) -> dict:
    """Creates custom ParagraphStyles supporting Greek Unicode."""
    base = getSampleStyleSheet()
    return {
        "CoverEyebrow": ParagraphStyle(
            "CoverEyebrow",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#2563eb"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "CoverTitle": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName=font_bold,
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "CoverSubtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#334155"),
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "H1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName=font_bold,
            fontSize=15.5,
            leading=21,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "H2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName=font_bold,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e3a8a"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=9.6,
            leading=14.2,
            textColor=colors.HexColor("#1e293b"),
            alignment=TA_JUSTIFY,
            spaceAfter=5,
        ),
        "BulletItem": ParagraphStyle(
            "BulletItem",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=9.4,
            leading=13.8,
            leftIndent=12,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=3.5,
        ),
        "Callout": ParagraphStyle(
            "Callout",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=9.3,
            leading=13.6,
            textColor=colors.HexColor("#065f46"),
            spaceAfter=4,
        ),
        "TableHeader": ParagraphStyle(
            "TableHeader",
            parent=base["Normal"],
            fontName=font_bold,
            fontSize=8.5,
            leading=11,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "TableCell": ParagraphStyle(
            "TableCell",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_LEFT,
        ),
        "TableCellCenter": ParagraphStyle(
            "TableCellCenter",
            parent=base["Normal"],
            fontName=font_reg,
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
        ),
        "Caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName=font_ital,
            fontSize=8.3,
            leading=11.5,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
    }


def make_styled_table(headers: list[str], rows: list[list[str]], col_widths: list[float], styles: dict) -> Table:
    """Builds a clean alternating-row ReportLab table with Greek paragraph wrapping."""
    data = [[Paragraph(f"<b>{h}</b>", styles["TableHeader"]) for h in headers]]
    for r in rows:
        row_cells = []
        for idx, cell in enumerate(r):
            st_use = styles["TableCell"] if idx in (0, len(r) - 1) else styles["TableCellCenter"]
            row_cells.append(Paragraph(str(cell), st_use))
        data.append(row_cells)

    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def make_speaker_box(text_html: str, styles: dict) -> Table:
    """Creates a green-bordered 'Τι να τονίσετε στην παρουσίαση' speaker note box."""
    p = Paragraph(text_html, styles["Callout"])
    box = Table([[p]], colWidths=[17.4 * cm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecfdf5")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#6ee7b7")),
                ("LINEBEFORE", (0, 0), (0, -1), 3.5, colors.HexColor("#10b981")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return box


def add_figure_if_exists(story: list, fig_name: str, caption: str, styles: dict, width_cm: float = 16.5, height_cm: float = 7.4):
    """Embeds a PNG figure from `reports/figures/` if present."""
    fig_path = FIGURES_DIR / fig_name
    if fig_path.exists():
        img = Image(str(fig_path), width=width_cm * cm, height=height_cm * cm, kind="proportional")
        story.append(Spacer(1, 3 * mm))
        story.append(KeepTogether([img, Paragraph(caption, styles["Caption"])]))
        story.append(Spacer(1, 2 * mm))


def append_cover_and_sidebar(story: list, styles: dict):
    """Adds the Cover Header, Executive Summary, and Left Sidebar Guide."""
    story.append(
        Paragraph(
            "ΕΓΧΕΙΡΙΔΙΟ ΠΑΡΟΥΣΙΑΣΗΣ & ΕΠΕΞΗΓΗΣΗΣ ΠΛΑΤΦΟΡΜΑΣ • VISION ZERO 2030–2035",
            styles["CoverEyebrow"],
        )
    )
    story.append(
        Paragraph(
            "Παρατηρητήριο Οδικής Ασφάλειας Ελλάδας: Μηχανική Μάθηση, Γεωχωρική Ανάλυση & Strategic Foresight",
            styles["CoverTitle"],
        )
    )
    story.append(
        Paragraph(
            "Πλήρης οδηγός παρουσίασης όλων των <b>7 Καρτελών (Views)</b> του διαδραστικού Dashboard, "
            "με αναλυτικά υπομνήματα όρων, περιεχόμενο κάθε διαγράμματος/πίνακα και έτοιμα σημεία ομιλίας.",
            styles["CoverSubtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.2, color=colors.HexColor("#2563eb"), spaceAfter=10))

    story.append(Paragraph("1. Ταυτότητα Έργου & Αριστερή Κάθετη Μπάρα Πλοήγησης (Sidebar)", styles["H1"]))
    story.append(
        Paragraph(
            "Η πλατφόρμα αποτελεί ένα ολοκληρωμένο σύστημα υποστήριξης αποφάσεων (Decision-Support & Foresight System) "
            "για την Οδική Ασφάλεια στις <b>13 Διοικητικές Περιφέρειες της Ελλάδας (NUTS-2)</b>. Βασίζεται αποκλειστικά "
            "σε <b>100% επίσημα δεδομένα της ΕΛΣΤΑΤ (Έρευνα SDT04, 2015–2025)</b> χωρίς καμία συνθετική εγγραφή "
            "(Zero Synthetic Data Policy), εμπλουτισμένα με ανεξάρτητες μελέτες πεδίου (ΕΜΠ, EU Trendline, ESRA3, ERA5, ΕΚΑΒ).",
            styles["Body"],
        )
    )

    sb_headers = ["Στοιχείο Αριστερού Μενού", "Τιμή / Περιεχόμενο", "Πώς το εξηγείτε στην παρουσίαση"]
    sb_rows = [
        [
            "<b>13 NUTS-2 Περιφέρειες</b>",
            "780 Μηνιαίες Παρατηρήσεις (2021–2025)",
            "Καλύπτει και τις 13 επίσημες Περιφέρειες της Ελλάδας επί 60 συνεχόμενους μήνες (13 × 60 = 780).",
        ],
        [
            "<b>RF Test ROC-AUC</b>",
            "0.8567 (PR-AUC: 0.8716)",
            "Το μοντέλο RandomForest προβλέπει με 85.7% ακρίβεια ποιες Περιφέρειες θα εμφανίσουν έξαρση σοβαρών τροχαίων το 2025.",
        ],
        [
            "<b>Vision Zero Όφελος</b>",
            "-566 Ατυχήματα KSI / έτος",
            "Πλήρης εφαρμογή των 5 αξόνων πολιτικής αποτρέπει 566 θανατηφόρα/σοβαρά ατυχήματα ετησίως (-25.9%).",
        ],
        [
            "<b>Ετήσιο Κοινωνικό Όφελος</b>",
            "€650.9 Εκατ. / έτος",
            "Βάσει του ευρωπαϊκού προτύπου SafetyCube (€1.15M ανά KSI), εξοικονομούνται €650.9M σε ΜΕΘ, περίθαλψη και ΑΕΠ.",
        ],
        [
            "<b>Ορισμός KSI</b>",
            "Killed or Seriously Injured",
            "Τροχαίο ατύχημα με τουλάχιστον 1 νεκρό (εντός 30 ημερών) ή 1 βαριά τραυματία. Ποσοστό KSI = (Ατυχήματα KSI / Σύνολο Ατυχημάτων) × 100.",
        ],
        [
            "<b>Το Ελληνικό Παράδοξο</b>",
            "Αττική (6.5% KSI) vs Νησιά/Επαρχία (20–25%+)",
            "Η Αττική έχει το 53.5% των ατυχημάτων αλλά χαμηλή βαρύτητα λόγω μποτιλιαρίσματος. Η περιφέρεια και τα νησιά έχουν 3×–4× θνησιμότητα ανά τροχαίο.",
        ],
    ]
    story.append(make_styled_table(sb_headers, sb_rows, [4.2 * cm, 3.8 * cm, 9.4 * cm], styles))
    story.append(Spacer(1, 4 * mm))
    story.append(
        make_speaker_box(
            "<b>🎤 Τι να πείτε ξεκινώντας την παρουσίαση:</b> «Η πλατφόρμα μας δεν δείχνει απλώς τι συνέβη στο παρελθόν. "
            "Συνδυάζει (1) Επίσημη Στατιστική ΕΛΣΤΑΤ, (2) Πρόγνωση Μηχανικής Μάθησης 12 μηνών, (3) Προσομοιωτή Παρεμβάσεων What-If, "
            "(4) Ανεξάρτητες Έρευνες Πεδίου ΕΜΠ/ΕΕ και (5) Στρατηγική Προόραση 10ετίας (2026–2035) σε ένα ενιαίο περιβάλλον.»",
            styles,
        )
    )
    story.append(PageBreak())


def append_tabs_1_to_4(story: list, styles: dict):
    """Adds comprehensive sections for Tabs 1, 2, 3, and 4."""
    # ---------------- TAB 1 ----------------
    story.append(
        Paragraph("Καρτέλα 1: 🗺️ Διαδραστικός Χάρτης & What-If Προσομοιωτής Μέτρων (2026)", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Αποτελεί την κεντρική οθόνη λήψης αποφάσεων. Επιτρέπει στον χρήστη να δει στον χάρτη της "
            "Ελλάδας πού εντοπίζεται ο κίνδυνος το 2026 και να προσομοιώσει σε πραγματικό χρόνο (What-If) πόσες ζωές και πόσα "
            "εκατομμύρια ευρώ σώζονται όταν ενεργοποιούμε από 0% έως 100% τους 5 άξονες του Ευρωπαϊκού Safe System.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Περιεχόμενο & Διαδραστικά Εργαλεία Καρτέλας 1:", styles["H2"]))
    for b in [
        "• <b>5 Κουμπιά Έτοιμων Σεναρίων (Presets):</b> (1) <i>Baseline 0%</i>, (2) <i>Αστική Ασπίδα 30 km/h</i>, (3) <i>Θερινό Blitz Νησιών & Μοτο</i>, (4) <i>Υπεραστικοί Άξονες 2+1</i>, και (5) <i>Πλήρες Vision Zero 2030</i>.",
        "• <b>5 Ρυθμιστές Έντασης Πολιτικής (Sliders 0%–100%):</b> Κάμερες Μέσης Ταχύτητας (Section Control), Κράνος & Νυχτερινό Αλκοτέστ, Στηθαία 2+1 Υπεραστικών Οδών, Ζώνες 30 km/h στις Πόλεις, και Κινητές Μονάδες ΕΚΑΒ (Golden Hour).",
        "• <b>4 Κάρτες Bento KPI:</b> Δείχνουν δυναμικά τα Αποτρεπόμενα KSI, τα Τελικά Αναμενόμενα KSI (από 2,186 Baseline), το Μέσο Περιφερειακό Ποσοστό KSI (%) και το Κοινωνικοοικονομικό Όφελος σε € Εκατ.",
        "• <b>Διαδραστικός Χάρτης NUTS-2 & Οριζόντιο Γράφημα Πριν/Μετά:</b> Το μέγεθος κύκλου δείχνει τον όγκο KSI και το χρώμα τη βαρύτητα KSI (%). Στο γράφημα στόχος είναι η πράσινη μπάρα να πέσει κάτω από το όριο συναγερμού 21%.",
    ]:
        story.append(Paragraph(b, styles["BulletItem"]))

    pkg_path = REPORTS_DIR / "policy_simulation_scenarios.csv"
    if pkg_path.exists():
        pkg_df = pd.read_csv(pkg_path)
        headers = ["Πακέτο Πολιτικής (Preset)", "Baseline KSI", "Τελικά KSI", "Αποτρεπόμενα KSI", "Μείωση (%)", "Όφελος (€M)"]
        rows = []
        for _, r in pkg_df.iterrows():
            rows.append([
                r["package_name"],
                f"{int(r['baseline_annual_ksi_accidents']):,}",
                f"{int(r['simulated_annual_ksi_accidents']):,}",
                f"<b>{int(r['annual_ksi_accidents_prevented']):,}</b>",
                f"-{r['annual_ksi_reduction_pct']:.1f}%",
                f"€{r['estimated_socioeconomic_benefit_eur_m']:.1f}M",
            ])
        story.append(Spacer(1, 2 * mm))
        story.append(make_styled_table(headers, rows, [5.4 * cm, 2.2 * cm, 2.2 * cm, 2.6 * cm, 2.3 * cm, 2.7 * cm], styles))

    add_figure_if_exists(
        story,
        "policy_counterfactual_impact.png",
        "Εικόνα 1.1: Αποτελέσματα Προσομοιωτή Παρεμβάσεων What-If (Αποτρεπόμενα KSI και Μείωση Βαρύτητας ανά Περιφέρεια).",
        styles,
        width_cm=16.5,
        height_cm=6.8,
    )
    story.append(
        make_speaker_box(
            "<b>🎤 Τι να τονίσετε στην Καρτέλα 1:</b> «Πατώντας το κουμπί <i>Πλήρες Vision Zero 2030</i>, βλέπουμε ότι ακόμη και μία "
            "συντηρητική μείωση 25.9% στη βαρύτητα των ατυχημάτων μεταφράζεται σε <b>566 λιγότερα θανατηφόρα/σοβαρά τροχαία "
            "μέσα σε ένα μόλις έτος</b> και εξοικονόμηση <b>€650.9 εκατ.</b> για το σύστημα υγείας και την οικονομία.»",
            styles,
        )
    )
    story.append(PageBreak())

    # ---------------- TAB 2 ----------------
    story.append(
        Paragraph("Καρτέλα 2: 🚨 Πρόγνωση 2026 & Σχέδιο Δράσης ανά Περιφέρεια", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Παρέχει μηνιαία έγκαιρη προειδοποίηση (12-Month Early Warning) για καθεμία από τις 13 "
            "Περιφέρειες το 2026 (156 προβλέψεις = 13 Περιφέρειες × 12 μήνες) και παράγει αυτόματα εξειδικευμένη επιχειρησιακή "
            "συνταγή μέτρων για την Τροχαία, το ΕΚΑΒ και τις τεχνικές υπηρεσίες.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Υπόμνημα Επιπέδων Συναγερμού (Risk Tiers) στην Καρτέλα 2:", styles["H2"]))
    tier_headers = ["Επίπεδο Συναγερμού", "Κριτήριο Ενεργοποίησης", "Επιχειρησιακή Σημασία & Προτεραιότητα"]
    tier_rows = [
        [
            "<b>🔴 CRITICAL SUMMER ALERT</b>",
            "Θερινοί μήνες (Ιούν–Σεπ), KSI ≥ 22% & Δίκυκλα ≥ 42%",
            "Κορύφωση κινδύνου σε Νότιο Αιγαίο, Ιόνια Νησιά, Κρήτη, Βόρειο Αιγαίο. Απαιτεί προ-τοποθέτηση ασθενοφόρων ΕΚΑΒ και νυχτερινά μπλόκα κράνους/αλκοόλ.",
        ],
        [
            "<b>🟠 HIGH SEVERITY RISK</b>",
            "Ποσοστό Βαρύτητας KSI ≥ 21% (εκτός θερινής αιχμής νησιών)",
            "Υπεραστικοί άξονες (Πελοπόννησος, Στερεά, Δυτική Ελλάδα, Ήπειρος, Θεσσαλία). Έμφαση σε κάμερες μέσης ταχύτητας και στηθαία 2+1.",
        ],
        [
            "<b>🔵 URBAN VOLUME FOCUS</b>",
            "Μητροπολιτικά κέντρα (Αστικό δίκτυο ≥ 75%)",
            "Αττική & Κεντρική Μακεδονία: Χαμηλό ποσοστό KSI (6.5%–11%), αλλά τεράστιος όγκος ατυχημάτων. Έμφαση σε ζώνες 30 km/h, διαβάσεις πεζών και κόκκινο σηματοδότη.",
        ],
        [
            "<b>🟡 MODERATE / WATCH</b>",
            "Ποσοστό Βαρύτητας KSI < 21%",
            "Τακτική προληπτική επιτήρηση ταχύτητας, ζώνης ασφαλείας και συντήρηση οδοστρώματος.",
        ],
    ]
    story.append(make_styled_table(tier_headers, tier_rows, [4.5 * cm, 4.5 * cm, 8.4 * cm], styles))
    add_figure_if_exists(
        story,
        "regional_forecast_heatmap.png",
        "Εικόνα 2.1: Μηνιαίος Χάρτης Θερμότητας (Heatmap) Πρόγνωσης KSI (%) για τις 13 Περιφέρειες το 2026.",
        styles,
        width_cm=16.5,
        height_cm=7.2,
    )
    story.append(
        make_speaker_box(
            "<b>🎤 Τι να τονίσετε στην Καρτέλα 2:</b> «Το Heatmap αποκαλύπτει την έντονη εποχικότητα: από τον Ιούνιο έως τον "
            "Σεπτέμβριο, οι νησιωτικές και τουριστικές περιφέρειες (Νότιο Αιγαίο, Ιόνιο, Κρήτη) ξεπερνούν το 22%–26% KSI. "
            "Η πολιτεία πρέπει να μετακινεί εποχικά δυνάμεις Τροχαίας και ΕΚΑΒ πριν ξεκινήσει το καλοκαίρι.»",
            styles,
        )
    )
    story.append(PageBreak())

    # ---------------- TAB 3 ----------------
    story.append(
        Paragraph("Καρτέλα 3: 📊 Επίσημα Στοιχεία ΕΛΣΤΑΤ (Ελλάδα 2015–2025)", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Παρουσιάζει την επίσημη ιστορική εξέλιξη της οδικής ασφάλειας στην Ελλάδα (2015–2024) "
            "και τεκμηριώνει στατιστικά το <b>Ελληνικό Χωρικό Παράδοξο (Greek Spatial Severity Paradox)</b> στο περιφερειακό "
            "πάνελ 2021–2025.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Βασικοί Ορισμοί ΕΛΣΤΑΤ (Υπόμνημα Καρτέλας 3):", styles["H2"]))
    for b in [
        "• <b>Ατύχημα με Παθόντες (Έρευνα SDT04):</b> Κάθε τροχαίο με τουλάχιστον 1 τραυματία ή νεκρό (δεν περιλαμβάνονται ατυχήματα μόνο με υλικές ζημιές).",
        "• <b>Νεκρός (Ορισμός 30 Ημερών):</b> Κάθε πρόσωπο που καταλήγει ακαριαία ή εντός 30 ημερών από το τροχαίο (πρότυπο ΕΕ / UNECE). Το 2024 καταγράφηκαν <b>665 νεκροί</b> και <b>546 βαριά τραυματίες</b> σε <b>11,312 ατυχήματα</b>.",
        "• <b>Το Χωρικό Παράδοξο Όγκου–Βαρύτητας:</b> Η Αττική συγκεντρώνει πάνω από τα μισά τροχαία της χώρας (53.5%) αλλά έχει το χαμηλότερο ποσοστό KSI (~6.5%), ενώ η Περιφέρεια έχει 3 έως 4 φορές υψηλότερη βαρύτητα (20%–25%+).",
    ]:
        story.append(Paragraph(b, styles["BulletItem"]))

    add_figure_if_exists(
        story,
        "elstat_regional_paradox_greece.png",
        "Εικόνα 3.1: Το Χωρικό Παράδοξο της ΕΛΣΤΑΤ — Όγκος Ατυχημάτων (Αριστερά) έναντι Βαρύτητας KSI % (Δεξιά).",
        styles,
        width_cm=16.5,
        height_cm=7.0,
    )

    # ---------------- TAB 4 ----------------
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph("Καρτέλα 4: 🤖 Αξιολόγηση Μοντέλων ML & Ablation Study (Test Set 2025)", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Αποδεικνύει την επιστημονική αυστηρότητα του συστήματος Μηχανικής Μάθησης. Χρησιμοποιούμε "
            "αυστηρό χρονολογικό διαχωρισμό (<b>2021–2023 Εκπαίδευση | 2024 Επικύρωση | 2025 Test Set</b>) ώστε να μην υπάρχει "
            "καμία διαρροή μελλοντικής πληροφορίας (Zero Data Leakage).",
            styles["Body"],
        )
    )
    bench_path = REPORTS_DIR / "benchmark_comparison.csv"
    if bench_path.exists():
        b_df = pd.read_csv(bench_path)
        headers = ["Μοντέλο Μηχανικής Μάθησης", "ROC-AUC (95% CI)", "PR-AUC (95% CI)", "F1-Score", "Precision", "Recall", "Brier Score"]
        rows = []
        for _, r in b_df.iterrows():
            rows.append([
                f"<b>{r['model']}</b>" if r["model"] == "RandomForest" else r["model"],
                f"{r['roc_auc']:.4f} {r['roc_auc_ci_95']}",
                f"{r['pr_auc']:.4f} {r['pr_auc_ci_95']}",
                f"{r['f1']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
                f"{r['brier_score']:.4f}",
            ])
        story.append(make_styled_table(headers, rows, [3.6 * cm, 3.3 * cm, 3.3 * cm, 1.7 * cm, 1.8 * cm, 1.7 * cm, 2.0 * cm], styles))

    story.append(Spacer(1, 2 * mm))
    story.append(
        Paragraph(
            "<b>Τι σημαίνουν οι μετρικές (Υπόμνημα Καρτέλας 4):</b> <b>ROC-AUC (0.8567)</b> = ικανότητα διαχωρισμού επικίνδυνων "
            "μηνών (0.50 = τυχαίο, 1.00 = τέλειο). <b>PR-AUC (0.8716)</b> = ακρίβεια εντοπισμού υψηλού KSI. "
            "<b>Recall (97.3%)</b> = το RandomForest εντόπισε το 97.3% των πραγματικά επικίνδυνων μηνών του 2025! "
            "<b>Brier Score (0.1552)</b> = το χαμηλότερο σφάλμα βαθμονόμησης πιθανοτήτων.",
            styles["Body"],
        )
    )
    story.append(PageBreak())


def append_tabs_5_to_7_and_qa(story: list, styles: dict):
    """Adds comprehensive sections for Tabs 5, 6, 7, and the Live Presentation Q&A Cheat-Sheet."""
    # ---------------- TAB 5 ----------------
    story.append(
        Paragraph("Καρτέλα 5: 🔍 Ερμηνευσιμότητα (Explainable AI) & Έλεγχος Περιφερειών", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Ανοίγει το «μαύρο κουτί» (Black Box) του αλγορίθμου RandomForest μέσω Permutation Feature "
            "Importance και ελέγχει αν οι πιθανότητες που δίνει το μοντέλο είναι δίκαιες και βαθμονομημένες (Calibrated) σε "
            "καθεμία από τις 13 Περιφέρειες.",
            styles["Body"],
        )
    )
    story.append(Paragraph("Λεξικό Μεταβλητών του Μοντέλου (Υπόμνημα Καρτέλας 5):", styles["H2"]))
    feat_headers = ["Όνομα Μεταβλητής στο Γράφημα", "Σπουδαιότητα (Δ PR-AUC)", "Επεξήγηση με Απλά Λόγια για την Παρουσίαση"]
    feat_rows = [
        [
            "<b>1. lag_12m_ksi_rate</b>",
            "+0.0253 (#1 Κορυφαία)",
            "Το ποσοστό KSI της ίδιας Περιφέρειας τον ίδιο μήνα του προηγούμενου έτους (εποχική μνήμη 12μήνου).",
        ],
        [
            "<b>2. urban_ratio</b>",
            "+0.0080 (#2 Κορυφαία)",
            "Ποσοστό ατυχημάτων εντός κατοικημένης περιοχής (διαχωρίζει τις μεγαλουπόλεις από το υπεραστικό δίκτυο).",
        ],
        [
            "<b>3. lag_12m_total_accidents</b>",
            "+0.0040 (#3 Κορυφαία)",
            "Ο συνολικός όγκος τροχαίων ατυχημάτων της Περιφέρειας 12 μήνες πριν.",
        ],
        [
            "<b>4. lag_1m_total_accidents</b>",
            "+0.0033 (#4 Κορυφαία)",
            "Ο συνολικός αριθμός ατυχημάτων τον αμέσως προηγούμενο μήνα (βραχυπρόθεσμη δυναμική).",
        ],
        [
            "<b>5. motorcycle_share</b>",
            "+0.0029 (#5 Κορυφαία)",
            "Μερίδιο εμπλοκής μοτοσικλετών στα τροχαία (καθοριστικός δείκτης ευάλωτων χρηστών οδού).",
        ],
    ]
    story.append(make_styled_table(feat_headers, feat_rows, [4.6 * cm, 3.2 * cm, 9.6 * cm], styles))
    add_figure_if_exists(
        story,
        "feature_importance_and_ablation.png",
        "Εικόνα 5.1: Σπουδαιότητα Χαρακτηριστικών (Permutation Importance) και Μελέτη Αφαίρεσης Ομάδων (Ablation Study).",
        styles,
        width_cm=16.5,
        height_cm=6.8,
    )
    story.append(PageBreak())

    # ---------------- TAB 6 ----------------
    story.append(
        Paragraph("Καρτέλα 6: 🔬 Ανεξάρτητες Έρευνες & Ευρωπαϊκά KPIs (ΕΜΠ / ESRA3 / ERA5)", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Επιλύει το μεθοδολογικό ζήτημα της κυκλικότητας (Zero Circularity). Επειδή οι πίνακες "
            "ατυχημάτων της Eurostat/CARE προέρχονται από την ίδια την ΕΛΣΤΑΤ, στην Καρτέλα 6 ενσωματώσαμε <b>100% ανεξάρτητες "
            "πρωτογενείς πηγές πεδίου</b> που μετρούν τη συμπεριφορά των οδηγών <i>πριν</i> συμβεί το ατύχημα, καθώς και "
            "τηλεματικά, κλιματικά και προνοσοκομειακά δεδομένα.",
            styles["Body"],
        )
    )
    ext_kpi_path = DATA_DIR / "independent_road_safety_kpis_greece.csv"
    if ext_kpi_path.exists():
        ek_df = pd.read_csv(ext_kpi_path)
        headers = ["Δείκτης Οδικής Ασφάλειας (KPI)", "Ελλάδα", "ΕΕ-27", "Απόκλιση", "Ανεξάρτητη Πρωτογενής Πηγή"]
        rows = []
        for _, r in ek_df.iterrows():
            rows.append([
                r["indicator_name_el"],
                f"<b>{r['greece_value_pct']:.1f}%</b>",
                f"{r['eu27_benchmark_pct']:.1f}%",
                f"{r['gap_vs_eu_pct']:+.1f}%",
                r["primary_source"],
            ])
        story.append(make_styled_table(headers, rows, [5.6 * cm, 1.8 * cm, 1.8 * cm, 2.0 * cm, 6.2 * cm], styles))

    story.append(Spacer(1, 3 * mm))
    story.append(
        make_speaker_box(
            "<b>🎤 Τι να τονίσετε στην Καρτέλα 6:</b> «Τα ανεξάρτητα δεδομένα πεδίου του ΕΜΠ και του ευρωπαϊκού προγράμματος "
            "Trendline εξηγούν <i>γιατί</i> τα νησιά και η επαρχία έχουν τόσο υψηλό KSI στην ΕΛΣΤΑΤ: η χρήση κράνους συνεπιβάτη "
            "είναι μόλις <b>65.2%</b> (έναντι 94.8% στην ΕΕ-27), η ζώνη στα πίσω καθίσματα <b>42.8%</b>, και ο διάμεσος χρόνος "
            "άφιξης του ΕΚΑΒ στα νησιά φτάνει τα <b>21–24.5 λεπτά</b> έναντι 11.5 λεπτών στην Αττική.»",
            styles,
        )
    )

    # ---------------- TAB 7 ----------------
    story.append(Spacer(1, 4 * mm))
    story.append(
        Paragraph("Καρτέλα 7: 🔭 Strategic Foresight 2026–2035 & Vision Zero Backcasting", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Μετατρέπει την πλατφόρμα σε εργαλείο <b>Στρατηγικής Προόρασης 10ετίας (2026–2035)</b>. "
            "Ενώ το Forecasting προβλέπει τους επόμενους 12 μήνες, το Foresight χαρτογραφεί <b>4 εναλλακτικά σενάρια "
            "μέλλοντος (Κώνος Αληθοφάνειας)</b>, 6 Μακρο-Τάσεις (Megatrends) και υπολογίζει μέσω <b>Backcasting</b> σε ποιο "
            "ακριβώς έτος κάθε Περιφέρεια πετυχαίνει τον στόχο της ΕΕ για <b>μείωση των KSI κατά 50%</b>.",
            styles["Body"],
        )
    )
    sc_headers = ["Σενάριο 10ετίας (2026–2035)", "Ετήσια KSI 2035", "Μεταβολή vs 2026", "Σωρευτικά KSI που Σώζονται (10ετία)", "Κοινωνικό Όφελος 10ετίας"]
    sc_rows = [
        [
            "<b>S1. Αδράνεια & Τουριστική/Κλιματική Πίεση</b>",
            "2,404 KSI/έτος",
            "+10.0% (Επιδείνωση)",
            "-1,134 (Αύξηση θυμάτων)",
            "-€1.30 Δισ. (Ζημία)",
        ],
        [
            "<b>S2. Ψηφιακή Επιτήρηση & Κάμερες AI</b>",
            "1,861 KSI/έτος",
            "-14.9%",
            "+2,115 KSI",
            "€2.43 Δισ.",
        ],
        [
            "<b>S3. Ανανέωση Στόλου ADAS & Υποδομές 2+1</b>",
            "1,562 KSI/έτος",
            "-28.5%",
            "+3,956 KSI",
            "€4.55 Δισ.",
        ],
        [
            "<b>S4. Πλήρες Vision Zero (Safe System)</b>",
            "<b>1,049 KSI/έτος</b>",
            "<b>-52.0% (Στόχος ΕΕ το 2034!)</b>",
            "<b>+7,390 KSI (8,524 vs S1)</b>",
            "<b>€8.50 Δισ.</b>",
        ],
    ]
    story.append(make_styled_table(sc_headers, sc_rows, [5.2 * cm, 2.5 * cm, 3.1 * cm, 3.6 * cm, 3.0 * cm], styles))
    add_figure_if_exists(
        story,
        "foresight_2026_2035_cone.png",
        "Εικόνα 7.1: Κώνος Αληθοφάνειας Strategic Foresight (2026–2035) και Σωρευτικά Αποτρεπόμενα KSI ανά Περιφέρεια.",
        styles,
        width_cm=16.5,
        height_cm=6.8,
    )
    story.append(PageBreak())

    # ---------------- TAB 8 ----------------
    story.append(
        Paragraph("Καρτέλα 8: 🧠 Κέντρο Λήψης Αποφάσεων (Human-in-the-Loop AI Copilot)", styles["H1"])
    )
    story.append(
        Paragraph(
            "<b>Σκοπός Καρτέλας:</b> Υλοποιεί την αρχιτεκτονική <b>Prescriptive Analytics με τον Άνθρωπο στον Βρόχο (EU AI Act "
            "Άρθρο 14 — Human Oversight)</b> για κρίσιμες επιχειρησιακές στιγμές. Ο αλγόριθμος συνθέτει τις προβλέψεις ML, τα "
            "δεδομένα πεδίου (κράνος, ταχύτητα, βροχόπτωση, χρόνος ΕΚΑΒ) και τον Επιχειρησιακό Κατάλογο 10 Παρεμβάσεων "
            "(<i>operational_interventions_catalog.csv</i>) για να εισηγηθεί <b>3 βέλτιστες επιλογές δράσης</b> υπό τους "
            "πραγματικούς περιορισμούς χρόνου και προϋπολογισμού της Διοίκησης.",
            styles["Body"],
        )
    )
    hitl_headers = ["Εισήγηση AI (Option)", "Στρατηγικός Στόχος", "Τι προτείνει στον Άνθρωπο-Λήπτη Απόφασης"]
    hitl_rows = [
        [
            "<b>🛡️ Επιλογή Α (Μέγιστη Ασφάλεια)</b>",
            "Μέγιστη Αποτροπή Θυμάτων KSI",
            "Αξιοποιεί πλήρως τον διαθέσιμο προϋπολογισμό (€K) επιλέγοντας τον συνδυασμό μέτρων που επιτυγχάνει τη μεγαλύτερη ποσοστιαία μείωση KSI.",
        ],
        [
            "<b>⚖️ Επιλογή Β (Βέλτιστο ROI)</b>",
            "Μέγιστη Σχέση Κόστους–Οφέλους",
            "Επιλέγει τις 3 πιο αποδοτικές δράσεις με τον υψηλότερο δείκτη ROI (κοινωνικό όφελος ανά ευρώ δαπάνης) εντός του 65% του προϋπολογισμού.",
        ],
        [
            "<b>⚡ Επιλογή Γ (Άμεση Κρίση 24–72h)</b>",
            "Ταχεία Κινητοποίηση Πεδίου",
            "Ενεργοποιεί αποκλειστικά δράσεις μηδενικής καθυστέρησης (μπλόκα αλκοτέστ/κράνους Τροχαίας, προ-τοποθέτηση ΕΚΑΒ, δυναμικά όρια VMS).",
        ],
    ]
    story.append(make_styled_table(hitl_headers, hitl_rows, [4.5 * cm, 4.2 * cm, 8.7 * cm], styles))
    story.append(Spacer(1, 3 * mm))
    story.append(
        make_speaker_box(
            "<b>🎤 Τι να τονίσετε στην Καρτέλα 8:</b> «Η Τεχνητή Νοημοσύνη δεν υποκαθιστά τον Διοικητή Τροχαίας, τον Συντονιστή "
            "ΕΚΑΒ ή τον Περιφερειάρχη. Του παρουσιάζει μέσα σε δευτερόλεπτα 3 κοστολογημένες, αιτιολογημένες εναλλακτικές "
            "(Επιλογές Α, Β, Γ), του επιτρέπει να προσθαφαιρέσει μέτρα και εκδίδει επίσημο <b>Πρακτικό Επιχειρησιακής Εντολής "
            "(Audit Trail)</b> με την ανθρώπινη υπογραφή και αιτιολόγηση.»",
            styles,
        )
    )
    story.append(Spacer(1, 4 * mm))

    # ---------------- Q&A CHEAT SHEET ----------------
    story.append(
        Paragraph("9. Σκονάκι Παρουσίασης: 5 Συχνές Ερωτήσεις & Έτοιμες Απαντήσεις", styles["H1"])
    )
    qa_items = [
        (
            "Ερώτηση 1: Γιατί προβλέπετε το Ποσοστό Βαρύτητας KSI (%) και όχι απλώς τον απόλυτο αριθμό ατυχημάτων;",
            "<b>Απάντηση:</b> «Επειδή στην Ελλάδα ισχύει το <i>Χωρικό Παράδοξο</i>: αν κοιτάζαμε μόνο τον απόλυτο αριθμό τροχαίων, "
            "όλοι οι πόροι θα πήγαιναν στην Αττική (53.5% των ατυχημάτων, αλλά μόλις 6.5% KSI λόγω χαμηλών ταχυτήτων). "
            "Μετρώντας το ποσοστό KSI (%), αναδεικνύουμε τις Περιφέρειες όπου 1 στα 4 τροχαία είναι θανατηφόρο ή βαρύ (Νότιο Αιγαίο, Κρήτη, Πελοπόννησος).»",
        ),
        (
            "Ερώτηση 2: Πώς εξασφαλίσατε ότι το μοντέλο Μηχανικής Μάθησης δεν «κλέβει» (Data Leakage);",
            "<b>Απάντηση:</b> «Εφαρμόσαμε αυστηρό χρονολογικό διαχωρισμό (Out-of-Time Split): το μοντέλο εκπαιδεύτηκε στα έτη "
            "2021–2023, ρυθμίστηκε στο 2024 και αξιολογήθηκε στο 2025. Επιπλέον, δεν χρησιμοποιήσαμε καμία μεταβλητή της ίδιας "
            "ημέρας/μήνα που γίνεται γνωστή μετά το ατύχημα, παρά μόνο χρονικές υστερήσεις (lags 1, 2, 3 και 12 μηνών).»",
        ),
        (
            "Ερώτηση 3: Τα δεδομένα από Ευρωπαϊκές βάσεις (CARE, ETSC) δεν ταυτίζονται με την ΕΛΣΤΑΤ;",
            "<b>Απάντηση:</b> «Πολύ σωστά! Οι πίνακες ατυχημάτων της CARE/Eurostat για την Ελλάδα τροφοδοτούνται από την έρευνα "
            "SDT04 της ΕΛΣΤΑΤ. Γι' αυτό στην Καρτέλα 6 προσθέσαμε αποκλειστικά <i>μη-ατυχηματικές</i> πρωτογενείς πηγές: "
            "μετρήσεις πεδίου χρήσης κράνους/ζώνης/ταχύτητας (EU Trendline & ΕΜΠ), έρευνα συμπεριφοράς ESRA3, τηλεματική "
            "(απότομα φρεναρίσματα), κλίμα (Copernicus ERA5) και χρόνους απόκρισης ΕΚΑΒ.»",
        ),
        (
            "Ερώτηση 4: Πώς υπολογίζονται οι μειώσεις ατυχημάτων στον Προσομοιωτή What-If;",
            "<b>Απάντηση:</b> «Χρησιμοποιούμε τους επίσημους <b>Συντελεστές Τροποποίησης Ατυχημάτων (Crash Modification Factors - CMFs)</b> "
            "του Ευρωπαϊκού Παρατηρητηρίου Οδικής Ασφάλειας (ERSO / SafetyCube) και του ΕΜΠ, σταθμισμένους ανάλογα με το προφίλ "
            "κάθε Περιφέρειας (π.χ. οι ζώνες 30 km/h αποδίδουν μέγιστα στην Αττική, ενώ τα στηθαία 2+1 στις υπεραστικές οδούς).»",
        ),
        (
            "Ερώτηση 5: Πότε πετυχαίνει η Ελλάδα τον ευρωπαϊκό στόχο μείωσης -50% των KSI;",
            "<b>Απάντηση:</b> «Σύμφωνα με το μοντέλο Backcasting στην Καρτέλα 7, με το σενάριο <i>Πλήρες Vision Zero (S4)</i> "
            "η Ελλάδα επιτυγχάνει τον στόχο <b>-50% KSI το έτος 2034</b> (και -52% το 2035), σώζοντας <b>7,390 σοβαρά/θανατηφόρα "
            "ατυχήματα</b> μέσα στη δεκαετία 2026–2035.»",
        ),
    ]
    for q, a in qa_items:
        story.append(Paragraph(f"❓ <b>{q}</b>", styles["H2"]))
        story.append(Paragraph(a, styles["Body"]))
        story.append(Spacer(1, 1.5 * mm))


def add_page_number(canvas, doc):
    """Draws a clean footer with page numbers on every page."""
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
    canvas.setLineWidth(0.5)
    canvas.line(1.8 * cm, 1.4 * cm, A4[0] - 1.8 * cm, 1.4 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(1.8 * cm, 0.95 * cm, "Greece Road Safety ML & Foresight Observatory — Presentation Guide")
    canvas.drawRightString(A4[0] - 1.8 * cm, 0.95 * cm, f"Page {doc.page}")
    canvas.restoreState()


def generate_pdf() -> list[Path]:
    """Generates the presentation guide PDF in both `docs/` and `reports/`."""
    font_reg, font_bold, font_ital = register_greek_fonts()
    styles = build_styles(font_reg, font_bold, font_ital)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    out_paths = [
        REPORTS_DIR / "Presentation_Guide_Road_Safety_Dashboard_GR.pdf",
    ]

    for out_path in out_paths:
        story = []
        append_cover_and_sidebar(story, styles)
        append_tabs_1_to_4(story, styles)
        append_tabs_5_to_7_and_qa(story, styles)

        doc = SimpleDocTemplate(
            str(out_path),
            pagesize=A4,
            leftMargin=1.8 * cm,
            rightMargin=1.8 * cm,
            topMargin=1.6 * cm,
            bottomMargin=1.8 * cm,
            title="Οδηγός Παρουσίασης Παρατηρητηρίου Οδικής Ασφάλειας Ελλάδας",
            author="Road Safety ML & Strategic Foresight Team",
        )
        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)

    return out_paths


if __name__ == "__main__":
    paths = generate_pdf()
    for p in paths:
        print(f"Generated PDF: {p} ({p.stat().st_size:,} bytes)")
