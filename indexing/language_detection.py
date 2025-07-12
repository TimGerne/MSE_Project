from lxml import html
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensures consistent results from langdetect
DetectorFactory.seed = 0

def detect_language_from_html(html_content: str) -> str:
    """
    Detects the primary language of an HTML document.

    The function uses a three-step approach:
    1. Attempts to read the `lang` attribute from the <html> tag.
    2. Searches for other elements with a `lang` attribute.
    3. Falls back to visible text and uses `langdetect` to determine language.

    Args:
        html_content (str): The raw HTML content as a string.

    Returns:
        str: The detected language code (e.g., 'en', 'de', 'fr').
             Returns 'unknown' if the language could not be reliably detected.
    """
    try:
        tree = html.fromstring(html_content)

        # 1. Check for lang attribute in <html>
        html_tag = tree.xpath('/html')[0]
        lang_attr = html_tag.attrib.get('lang')
        if lang_attr:
            return lang_attr.lower().split('-')[0]  # Normalize: 'en-US' -> 'en'
        
        # 2. Look for other lang attributes, for example in <head>
        lang_elements = tree.xpath('//*[@lang]')
        for el in lang_elements:
            lang_attr = el.get('lang')
            if lang_attr:
                return lang_attr.lower().split('-')[0]

        # 3. Fallback: Extract visible text and use langdetect
        text = tree.text_content()
        if not text.strip():
            return 'unknown'

        detected_lang = detect(text)
        return detected_lang

    except (IndexError, LangDetectException):
        return 'unknown'
