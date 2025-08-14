import os
import logging
from typing import Optional
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class TextExtractor:
    """
    Extract text from a local TXT file or from a URL.
    """
    def __init__(self, source: str):
        self.source = source
        self.is_url = source.lower().startswith(('http://', 'https://'))
        if not self.is_url and not os.path.isfile(source):
            logger.error(f"File not found: {source}")
            raise FileNotFoundError(f"File not found: {source}")

    def extract_text(self, preview: Optional[int] = None) -> str:
        """
        Extract text from TXT file or URL, with optional preview length.
        """
        logger.info(f"Starting extraction for {self.source} (type: {'URL' if self.is_url else 'TXT'})")
        try:
            if self.is_url:
                result = self._extract_url(preview)
            else:
                result = self._extract_txt(preview)
            logger.info(f"Extraction successful for {self.source}")
            return result
        except Exception as e:
            logger.error(f"Failed to extract text from {self.source}: {str(e)}")
            return f"[Error] Failed to extract text from {self.source}: {str(e)}"

    def _extract_txt(self, preview: Optional[int] = None) -> str:
        with open(self.source, 'r', encoding='utf-8') as f:
            return f.read(preview) if preview is not None else f.read()

    def _extract_url(self, preview: Optional[int] = None) -> str:
        response = requests.get(self.source)
        response.raise_for_status()
        text = response.text
        return text if preview is None else text[:preview]

# if __name__ == "__main__":
#     # # for text file ----------------------------------------
#     # txt_file = "/Users/abhishek/Desktop/ragTecorbAI/dummJson.txt"
#     # extractor_file = TextExtractor(txt_file)
#     # print(extractor_file.extract_text(preview=50))
#     # for url----------------------------------------
#     url = "https://dummyjson.com/users"
#     extractor_url = TextExtractor(url)
#     print(extractor_url.extract_text(preview=50))   
