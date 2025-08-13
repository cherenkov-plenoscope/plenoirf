import gzip
import json_line_logger
import io


class LoggerAppender:
    def __init__(self, payload, mode="b|gz"):
        self.strio = io.StringIO()
        self.mode = mode
        if "|gz" in self.mode:
            payload = gzip.decompress(payload)
        if "b" in self.mode:
            self.strio.write(bytes.decode(payload))
        else:
            assert "t" in mode, "Either mode 't' or 'b'."
            self.strio.write(payload)
        self.logger = json_line_logger.LoggerStream(stream=self.strio)

    def __enter__(self):
        return self

    def get_payload(self):
        self.strio.seek(0)
        out = self.strio.read()
        if "b" in self.mode:
            out = str.encode(out)
        if "|gz" in self.mode:
            out = gzip.compress(out)
        return out

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def __repr__(self):
        return f"{self.__class__.__name__:s}()"
