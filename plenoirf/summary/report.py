import pylatex
import tempfile
import os
import importlib.resources
import datetime


def noesc(text):
    return pylatex.utils.NoEscape(text)


def _document_config():
    return {
        "documentclass": "article",
        "document_options": [],
        "geometry_options": _document_geometry_options(),
        "font_size": "small",
        "page_numbers": True,
    }


def _document_geometry_options():
    return {
        "paper": "a4paper",
        # "paperwidth": "18cm",
        # "paperheight": "32cm",
        "head": "0cm",
        "left": "2cm",
        "right": "2cm",
        "top": "0cm",
        "bottom": "2cm",
        "includehead": True,
        "includefoot": True,
    }


class WorkDir:
    def __init__(self, path=None, suffix=None):
        if path is None:
            self.handle = tempfile.TemporaryDirectory(suffix=suffix)
            self.path = self.handle.name
            self.is_temporary = True
        else:
            self.handle = None
            self.path = path
            self.is_temporary = False

    def cleanup(self):
        if self.is_temporary:
            self.handle.cleanup()


class Report:
    def __init__(self, path, subtitle, bibliography_path=None, work_dir=None):
        self.path = path
        self.subtitle = subtitle
        if bibliography_path:
            self.bibliography_path = bibliography_path
        else:
            self.bibliography_path = _guess_bibliography_path()

        self.work_dir = WorkDir(path=path)

        self.tex = pylatex.Document(
            default_filepath=self.path,
            **_document_config(),
        )
        self.tex.preamble.append(pylatex.Package("multicol"))
        self.tex.preamble.append(pylatex.Package("lipsum"))
        self.tex.preamble.append(pylatex.Package("float"))
        self.tex.preamble.append(pylatex.Package("verbatim"))

        self.tex.preamble.append(
            noesc(
                r"\title{\Large Simulating the Cherenkov Plenoscope\\ \large "
                + self.subtitle
                + "}"
            )
        )
        self.tex.preamble.append(
            noesc(
                r"\author{\normalsize Sebastian A. Mueller and Werner Hofmann}"
            )
        )
        self.tex.preamble.append(
            noesc(r"\date{\normalsize " + date_now() + "}")
        )
        self.tex.append(noesc(r"\maketitle"))
        self.tex.append(noesc(r"\begin{multicols}{2}"))

    def __enter__(self):
        return self

    def finalize(self):
        self.tex.append(noesc(r"\bibliographystyle{apalike}"))
        self.tex.append(
            noesc(r"\bibliography{" + self.bibliography_path + "}")
        )

        self.tex.append(noesc(r"\end{multicols}{2}"))
        self.tex.generate_pdf(
            filepath=_strip_pdf_extension(path=self.path),
            clean_tex=False,
        )
        self.work_dir.cleanup()

    def __exit__(self, exc_type, exc_value, traceback):
        self.finalize()

    def __repr__(self):
        return (
            f"{__name__:s}.{self.__class__.__name__:s}(path='{self.path:s}')"
        )


def _strip_pdf_extension(path):
    if str.endswith(path.lower(), ".pdf"):
        return path[:-4]
    else:
        return path


def _guess_bibliography_path():
    starter_kit_dir = _guess_starter_kit_dir()
    return os.path.join(starter_kit_dir, "resources", "references.bib")


def _guess_starter_kit_dir():
    path = importlib.resources.files("plenoirf")
    path, end = os.path.split(path)
    assert end == "plenoirf"
    path, end = os.path.split(path)
    assert end == "plenoirf"
    path, end = os.path.split(path)
    assert end == "packages", (
        "This will only work when plenoirf was "
        "installed in the 'stater_kit' directory."
    )
    return path


def date_now():
    dt = datetime.datetime.now()
    return dt.isoformat()[0:16]
