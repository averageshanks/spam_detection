import os
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import pytesseract
from html2text import HTML2Text
from email.parser import BytesParser
from email.policy import default
from email.utils import parsedate_to_datetime

def parse_email(eml_file):
    image_text = ''
    html_text = ''

    with eml_file:
        parsed = BytesParser(policy=default).parse(eml_file)

    if parsed.is_multipart():
        print("Multiple parts")
        for part in parsed.walk():
            ctype = part.get_content_type()
            print(ctype)
            cdispo = str(part.get_content_disposition())
            if ctype == "text/plain" and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)
            else:
                body = ''
            if ctype == "image/png":
                encoded_data = part.get_payload()
                decoded_data = base64.b64decode(encoded_data)
                image = Image.open(BytesIO(decoded_data))
                image_array = np.array(image)
                image_text = pytesseract.image_to_string(image_array)
            if ctype == "text/html":
                html2text_converter = HTML2Text()
                html_text_payload = part.get_payload()
                html_text = html2text_converter.handle(html_text_payload)
    else:
        print("Single Part")
        body = parsed.get_payload(decode=True)

    email_data = {
        # header components
        "Unique-ID": format(parsed['Message-ID']),
        "To": format(parsed['to']),
        "From": format(parsed['from']),
        "Cc": format(parsed['Cc']),
        "Bcc": format(parsed['Bcc']),
        "Mail_User_Agent": format(parsed['']),
        "Date": parsedate_to_datetime(format(parsed['date'])),
        "Subject": format(parsed['subject']),

        # body component
        "Body": body,
        "Image_text": image_text,
        "HTML_text": html_text
    }

    return email_data
