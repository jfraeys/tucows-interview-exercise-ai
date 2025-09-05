from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

# Define the PDF path
pdf_path = "tests/data/domain_support_docs.pdf"

# Styles
styles = getSampleStyleSheet()
title_style = styles["Heading1"]
section_style = styles["Heading2"]
subsection_style = styles["Heading3"]
normal_style = styles["BodyText"]

# Content (with sections & subsections)
content = []

# Title
content.append(Paragraph("Domain Provider Support Documentation", title_style))
content.append(Spacer(1, 12))

# Section 1: Domain Suspension & Reactivation
content.append(Paragraph("1. Domain Suspension & Reactivation", section_style))

content.append(Paragraph("1.1 Reasons for Suspension", subsection_style))
content.append(
    Paragraph(
        "Domains may be suspended for the following reasons: "
        "policy violations, missing WHOIS information, or unpaid invoices.",
        normal_style,
    )
)

content.append(Paragraph("1.2 Reactivation Process", subsection_style))
content.append(
    Paragraph(
        "To reactivate your domain, update WHOIS details and resolve any outstanding invoices. "
        "For suspensions due to abuse or policy violations, contact the Abuse Team.",
        normal_style,
    )
)
content.append(Spacer(1, 12))

# Section 2: API Authentication Issues
content.append(Paragraph("2. API Authentication Issues", section_style))

content.append(Paragraph("2.1 Common Authentication Errors", subsection_style))
content.append(
    Paragraph(
        "Check that you are using the correct API key, that it has not expired, "
        "and that you are making requests over HTTPS with proper headers.",
        normal_style,
    )
)

content.append(Paragraph("2.2 Regenerating API Keys", subsection_style))
content.append(
    Paragraph(
        "If issues persist, generate a new API key in the Developer Portal and update your integrations accordingly.",
        normal_style,
    )
)
content.append(Spacer(1, 12))

# Section 3: Domain Transfer Support
content.append(Paragraph("3. Domain Transfer Support", section_style))

content.append(Paragraph("3.1 Preparing for Transfer", subsection_style))
content.append(
    Paragraph(
        "Unlock your domain in the dashboard and request an authorization (EPP) code. "
        "Ensure WHOIS contact information is accurate before starting the transfer.",
        normal_style,
    )
)

content.append(Paragraph("3.2 Restrictions", subsection_style))
content.append(
    Paragraph(
        "Domains cannot be transferred within 60 days of registration or a previous transfer.",
        normal_style,
    )
)
content.append(Spacer(1, 12))

# Section 4: Billing & Invoices
content.append(Paragraph("4. Billing & Invoices", section_style))

content.append(Paragraph("4.1 Paying Invoices", subsection_style))
content.append(
    Paragraph(
        "Invoices must be paid before renewal deadlines to avoid service interruptions. "
        "Payments can be made in the Billing section of your dashboard.",
        normal_style,
    )
)

content.append(Paragraph("4.2 Disputed Charges", subsection_style))
content.append(
    Paragraph(
        "If you believe you were charged incorrectly, open a billing support ticket for review.",
        normal_style,
    )
)
content.append(Spacer(1, 12))

# Build PDF
doc = SimpleDocTemplate(pdf_path, pagesize=LETTER)
doc.build(content)

pdf_path
