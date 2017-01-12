## Data
- should integrate the quandl code written for ee103. no need to explicitly call Arctic. we can just load in memory.
- should include a list of assets into the package. at a minimum the SP500 and about 100 ETFs.
in addition to what i have it would have a column identifying the type of asset
(stock, etf, rate)

## Results
- fix the current `__repr__`, it's broken. we should give very generalize infos if we implement a repr at alpha_model
- only support minimal set of operations. maybe prune it after the
examples are done
- naming should be iron-clad here, match the paper exactly

## Benchmark
- in retrospect it was stupid to put it in so many places. we should have it only where it's needed.
- **risk**. it should be in `BaseRisk` and evaluate in the expression,
so we can forget about it in the rest of the code
- **constraints** on a as-needed basis, i guess. or again in the base class
- **returns**. we don't need it there
- **costs** neither!

## Tests
- remove functionalities we don't test (e.g. a lot of results)

## Kelly
- it's a secondary objective
- make a separate module that implements
 - a Kelly "returns" object (it won't be called `AlphaStream` any more)
 - a Kelly constraint that does the risk part
 - that's pretty much it. we must also have a `**kwargs` into the policy object so we can pass solver options (max iters for SCS)
