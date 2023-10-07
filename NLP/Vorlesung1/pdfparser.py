from PyPDF4 import PdfFileReader, PdfFileWriter

def extract_pages(pdf_path):
    pdfFile = PdfFileReader(pdf_path, 'rb')
    pages = PdfFileReader.pages
    text = pages.extract_text()
    print(text)
    return

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

def extract_pages_miner(pdf_path):
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                print(element.get_text())
    return

if __name__ == "__main__":
    extract_pages('1706.03762')
    extract_pages_miner('1706.03762')
    # TODO