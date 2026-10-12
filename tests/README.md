# Tests

Install the development dependencies and run the offline suite:

```bash
python -m pip install -e '.[dev,simulation]'
python -m pytest -q
```

The suite covers configuration, demonstration preparation, model-provider
interfaces, recording and the RoboDojo action/evaluation contract. Device and
provider tests use mocks or temporary files; they do not start real hardware,
Isaac Sim or paid model calls.

Run only the simulation tests with:

```bash
python -m pytest -q tests/simulation
```

For a native simulator smoke test, follow the [RoboDojo guide](../docs/robodojo.md).
