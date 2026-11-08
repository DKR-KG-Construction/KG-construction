"""Checks the real HTTP backends against a tiny fake server (request shape + response parsing).
This does NOT test any model; it only proves the client code talks to Ollama / OpenAI-style APIs correctly."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from kgrepair.llm import BackendError, OllamaBackend, OpenAICompatBackend, make_backend

SEEN = {}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN[self.path] = (body, dict(self.headers))
        if self.path == "/api/chat":
            out = {"message": {"content": '{"entities":[],"triples":[]}'}, "prompt_eval_count": 11, "eval_count": 7}
        else:
            out = {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_request_and_parse(server):
    b = OllamaBackend({"model": "m", "host": server, "temperature": 0, "seed": 3, "max_tokens": 99, "num_ctx": 1234})
    schema = {"type": "object"}
    r = b.generate("sys", "usr", schema)
    body, _ = SEEN["/api/chat"]
    assert body["model"] == "m" and body["stream"] is False and body["format"] == schema
    assert body["options"] == {"temperature": 0, "seed": 3, "num_predict": 99, "num_ctx": 1234}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert r["usage"] == {"prompt_tokens": 11, "completion_tokens": 7} and r["text"].startswith("{")


def test_openai_compat_request_and_parse(server, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret")
    b = OpenAICompatBackend({"model": "m", "host": server, "api_key_env": "MY_KEY", "max_tokens": 50})
    r = b.generate("sys", "usr", {"type": "object"})
    body, headers = SEEN["/chat/completions"]
    assert body["response_format"]["type"] == "json_schema" and body["max_tokens"] == 50
    assert headers["Authorization"] == "Bearer secret"
    assert r["usage"] == {"prompt_tokens": 5, "completion_tokens": 3}


def test_placeholder_model_is_rejected():
    with pytest.raises(BackendError):
        make_backend({"backend": "ollama", "model": "CHANGE_ME", "host": "http://x"})
