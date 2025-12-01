import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

class ValidationPackGenerator:
    def __init__(self, output_dir="docs/validation_pack"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        self.title_style = self.styles['Heading1']
        self.normal_style = self.styles['Normal']
        
    def generate_all(self):
        print(f"Generating Validation Pack in {self.output_dir}...")
        self.generate_gap_analysis()
        self.generate_iq_protocol()
        self.generate_oq_protocol()
        self.generate_pq_protocol()
        self.generate_traceability_matrix()
        print("Validation Pack generation complete.")

    def _create_pdf(self, filename, title, content_elements):
        doc = SimpleDocTemplate(
            os.path.join(self.output_dir, filename),
            pagesize=letter,
            rightMargin=72, leftMargin=72,
            topMargin=72, bottomMargin=18
        )
        
        story = []
        # Header
        story.append(Paragraph("POLYMORPH-LITE v5.0", self.styles['Title']))
        story.append(Paragraph(title, self.styles['Heading2']))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.normal_style))
        story.append(Spacer(1, 0.5*inch))
        
        story.extend(content_elements)
        
        doc.build(story)

    def generate_gap_analysis(self):
        data = [
            ["21 CFR Part 11 Requirement", "POLYMORPH Implementation", "SOP Requirement"],
            ["11.10(a) Validation", "Automated IQ/OQ/PQ scripts", "Execute & Sign Validation Protocols"],
            ["11.10(b) Copies of Records", "PDF Export, JSON Audit Trail", "Regular Backup Verification"],
            ["11.10(c) Protection of Records", "Database Encryption, RBAC", "Manage User Access Levels"],
            ["11.10(d) Limiting System Access", "JWT Auth, Unique User IDs", "Password Policy Enforcement"],
            ["11.10(e) Audit Trails", "Immutable DB Log, SHA-256 Hash", "Periodic Audit Trail Review"],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 2.0*inch, 2.0*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        self._create_pdf("Gap_Analysis.pdf", "21 CFR Part 11 Gap Analysis", [table])

    def generate_iq_protocol(self):
        elements = [
            Paragraph("1. Installation Verification", self.styles['Heading3']),
            Paragraph("Verify that all software components are installed correctly.", self.normal_style),
            Spacer(1, 0.2*inch),
            Paragraph("Test Steps:", self.styles['Heading4']),
            Paragraph("1. Check Docker container status (docker ps)", self.normal_style),
            Paragraph("2. Verify database connection", self.normal_style),
            Paragraph("3. Verify API endpoint availability", self.normal_style),
        ]
        self._create_pdf("IQ_Protocol.pdf", "Installation Qualification (IQ) Protocol", elements)

    def generate_oq_protocol(self):
        elements = [
            Paragraph("1. Operational Verification", self.styles['Heading3']),
            Paragraph("Verify that the system operates according to functional specifications.", self.normal_style),
            Spacer(1, 0.2*inch),
            Paragraph("Test Steps:", self.styles['Heading4']),
            Paragraph("1. Run 'SimDAQ' acquisition workflow", self.normal_style),
            Paragraph("2. Verify data integrity in database", self.normal_style),
            Paragraph("3. Test user login and permission denial", self.normal_style),
        ]
        self._create_pdf("OQ_Protocol.pdf", "Operational Qualification (OQ) Protocol", elements)

    def generate_pq_protocol(self):
        elements = [
            Paragraph("1. Performance Verification", self.styles['Heading3']),
            Paragraph("Verify that the system performs consistently under load.", self.normal_style),
            Spacer(1, 0.2*inch),
            Paragraph("Test Steps:", self.styles['Heading4']),
            Paragraph("1. Execute 'Tier-1 Raman Sweep' workflow 10 times", self.normal_style),
            Paragraph("2. Verify AI confidence scores > 0.85", self.normal_style),
            Paragraph("3. Confirm audit trail completeness", self.normal_style),
        ]
        self._create_pdf("PQ_Protocol.pdf", "Performance Qualification (PQ) Protocol", elements)

    def generate_traceability_matrix(self):
        data = [
            ["URS ID", "Requirement", "FRS ID", "Test Protocol"],
            ["URS-01", "Secure User Login", "FRS-AUTH-01", "OQ-03"],
            ["URS-02", "Acquire Raman Spectra", "FRS-DRV-01", "OQ-01"],
            ["URS-03", "Audit Trail", "FRS-AUDIT-01", "PQ-03"],
            ["URS-04", "AI Analysis", "FRS-AI-01", "PQ-02"],
        ]
        
        table = Table(data, colWidths=[1.0*inch, 2.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        self._create_pdf("Traceability_Matrix.pdf", "Traceability Matrix", [table])

if __name__ == "__main__":
    generator = ValidationPackGenerator()
    generator.generate_all()
