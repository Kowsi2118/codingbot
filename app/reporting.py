from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, Optional

import jinja2

try:
    import pdfkit  # type: ignore
except Exception:  # pragma: no cover
    pdfkit = None  # type: ignore


TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def render_report_html(context: Dict[str, Any], template_name: str = "report.html") -> str:
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)), autoescape=True
    )
    template = env.get_template(template_name)
    return template.render(**context)


def html_to_pdf_bytes(html: str) -> Optional[bytes]:
    if pdfkit is None:
        return None
    try:
        return pdfkit.from_string(html, False)  # type: ignore
    except Exception:
        return None


def to_download_link(data: bytes, filename: str, mime: str) -> str:
    b64 = base64.b64encode(data).decode()
    return f'<a download="{filename}" href="data:{mime};base64,{b64}">Download {filename}</a>'