"""Offline persistence fixture; this callable is not a language model."""
import os
import nsn


def fixture(prompt):
    return '8080' if '<retrieved_evidence>' in prompt and '8080' in prompt else 'acknowledged'


def main():
    directory = os.environ.get('NSN_DATA_DIR', './agent-memory')
    namespace = os.environ.get('NSN_NAMESPACE', 'restart_demo')
    nsn.init(directory)
    model = nsn.wrap(fixture, namespace=namespace)
    model('The payment gateway port is 8080.')
    nsn.close()
    nsn.init(directory)
    model = nsn.wrap(fixture, namespace=namespace)
    answer = model('What is the payment gateway port?')
    nsn.close()
    assert answer == '8080', 'Prior evidence did not survive restart'
    print('Offline persistence fixture passed: gateway port 8080')


if __name__ == '__main__':
    main()
