"""Read physical JSONL records without splitting Unicode string content."""
import json
from pathlib import Path


def parse_rows(raw):
    records=raw.split(b'\n')
    if records[-1]==b'':records.pop()
    return [json.loads(record.decode('utf-8')) for record in records]


def read_rows(path):
    return parse_rows(Path(path).read_bytes())
