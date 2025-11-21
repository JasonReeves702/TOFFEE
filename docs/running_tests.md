## Running the Test Functions

To run the test files in TOFFEE using pytest import toffee and pytest

```python
import toffee
import pytest

```
Then run pytest on the test files by opening up terminal and running

```python
!pytest --pyargs toffee.tests
```

It should take about 10-20 seconds to run. The results should either return no errors and no warning if you use a version of numpy older than numpy 2.0.0 or should return 68 warning about the depreciation of np.trapz if you're using a verion older than numpy 2.0.0. If you don't have oktopus installed on your machine you will also recieve a warning about stemming from its dependency for lightkurve.

If you receieve and error or any other warning please open an issue and report it.
