"""Decode native edits or the equivalent JSON Patch for the one editable text slot."""
import json


def normalize(output, current):
    text = output.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2].rsplit("```", 1)[0]
    data = json.loads(text)
    if isinstance(data, dict) and set(data) == {"edits"}:
        edits = data["edits"]
        if not isinstance(edits, list) or len(edits) != 1 or not isinstance(edits[0], dict):
            raise ValueError("exactly one native edit object required")
        edit = edits[0]
        required = {"op", "content"} if edit.get("op") == "append" else {"op", "target", "content"}
        if set(edit) != required or edit["op"] not in ("append", "replace"):
            raise ValueError("expected native append or replace")
        if edit["op"] == "replace" and edit["target"] != current:
            raise ValueError("replace must target the current instruction exactly")
        return data, "native"
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        edit = data[0]
        if set(edit) == {"op", "path", "value"} and edit["op"] == "replace" and edit["path"] == "/0/content":
            if not isinstance(edit["value"], str) or not edit["value"].strip():
                raise ValueError("empty replacement")
            return {"edits": [{"op": "replace", "target": current, "content": edit["value"]}]}, "json_patch"
    raise ValueError("one native edits envelope or one replace of /0/content required")
