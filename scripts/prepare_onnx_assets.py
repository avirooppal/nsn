"""Explicit preparation of pinned MiniLM ONNX assets; never run on SDK import."""
import argparse
import hashlib
from pathlib import Path
import urllib.request

REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
FILES = {
    'model.onnx': ('onnx/model.onnx', '6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452'),
    'tokenizer.json': ('tokenizer.json', 'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037'),
    'config.json': ('config.json', '953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41'),
    'sentence_bert_config.json': ('sentence_bert_config.json', 'fc1993fde0a95c24ec6c022539d41cf6e2f7c9721e5415d6fb6897472a9cd4b7'),
    '1_Pooling/config.json': ('1_Pooling/config.json', '4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23'),
    'README.md': ('README.md', 'dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0'),
}


def prepare(target):
    target = Path(target)
    for name, (remote, expected) in FILES.items():
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError('Existing asset checksum mismatch: ' + name)
            continue
        partial = destination.with_suffix(destination.suffix + '.part')
        url = 'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/' + REVISION + '/' + remote
        checksum = hashlib.sha256()
        with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                checksum.update(chunk)
                handle.write(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError('Downloaded asset checksum mismatch: ' + name)
        partial.replace(destination)
    return str(target.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    print(prepare(parser.parse_args().target))
