from fpdf import FPDF


class PDFHelper(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, "Used Vehicles Database and Specifications", 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 5, title, 0, 1, 'L')
        self.ln(3)

    def chapter_body(self, body):
        self.set_font('Arial', '', 9)
        self.multi_cell(0, 5, body)
        self.ln(1)

    def add_chapter(self, title, body):
        self.add_page()
        self.chapter_title(title)
        self.chapter_body(body)
