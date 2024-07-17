from fpdf import FPDF


class PDFHelper(FPDF):
    def chapter_body(self, body):
        self.set_font('Arial', '', 9)
        self.multi_cell(0, 5, body)
        self.ln(1)

    def add_chapter(self, body):
        self.add_page()
        self.chapter_body(body)
