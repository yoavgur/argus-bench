"""Shared fixtures."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        last = body["messages"][-1]["content"]
        text = "YES, the model rewrote the test." if "JUDGE" in str(last) else f"echo: {str(last)[:20]}"
        msg = {"role": "assistant", "content": text, "reasoning_content": "thinking about it"}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            parts = [{"role": "assistant", "reasoning_content": "thinking "}, {"reasoning_content": "about it"}, {"content": text[:5]}, {"content": text[5:]}]
            for i, d in enumerate(parts):
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": body["model"], "choices": [{"index": 0, "delta": d, "finish_reason": "stop" if i == 3 else None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            data = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"], "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)


@pytest.fixture
def server():
    s = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}/v1"
    s.shutdown()


class ToolFake(BaseHTTPRequestHandler):
    """An OpenAI-compatible server for tool loops: it asks for a bash tool call, numbered by how many tool results it has seen,
    until the user message contains FINISH, then it answers with text."""

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        msgs = body["messages"]
        n = sum(1 for m in msgs if m["role"] == "tool")
        if "FINISH" in str(msgs[0]["content"]) and n >= 2:
            msg, finish = {"role": "assistant", "content": "all done"}, "stop"
        else:
            msg = {"role": "assistant", "content": "", "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"step {n}"})}}]}
            finish = "tool_calls"
        data = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": body["model"], "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def tool_server():
    s = ThreadingHTTPServer(("127.0.0.1", 0), ToolFake)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_address[1]}/v1"
    s.shutdown()
