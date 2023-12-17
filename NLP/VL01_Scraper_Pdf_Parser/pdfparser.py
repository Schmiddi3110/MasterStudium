from PyPDF4 import PdfFileReader
from pdfminer.high_level import extract_pages

def extract_pages(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf = PdfFileReader(f)
        information = pdf.pages
        with open("PdfContent.txt", "w") as textfile:
            for page in information:
                textfile.write(page.extractText())
    return

if __name__ == "__main__":
    extract_pages("Paper.pdf")
