## Examples
- [ ] fix dataset for all three examples, or be prepared to gimnick through parameters if we change it later
- [ ] for SPO decide which plot/tables we want to produce. it will probably be a simulation taking ~1hr. you can run it on desktop at home:
  - [ ] three values of tcost par. 3 values of risk par. plot the 3 curves (same tcost par is same curve) on risk/reward plot. you should be careful, what is return? (mean growth rate?) and what is volatility? look in paper for answers


## Data
- [ ] should integrate the quandl code written for ee103. no need to explicitly call Arctic. we can just load in memory.
- [ ] should include a list of assets into the package. at a minimum the SP500 and about 100 ETFs.
in addition to what i have it would have a column identifying the type of asset
(stock, etf, rate)

## Results
- [x] fix the current `__repr__`, it's broken. we should give very generalize infos if we implement a repr at alpha_model
- [x] only support minimal set of operations. maybe prune it after the
examples are done
- [ ] naming should be iron-clad here, match the paper exactly

## Benchmark
In retrospect it was stupid to put it in so many places. we should have it only where it's needed.
- [x] **risk**. it should be in `BaseRisk` and evaluate in the expression,
so we can forget about it in the rest of the code
- [x] **constraints** on a as-needed basis, i guess. or again in the base class
- [x] **returns**. we don't need it there
- [x] **costs** neither!

## Tests
- [ ] remove functionalities we don't test (e.g. a lot of results)
- [ ] figure out why nosetests gives C error on MKL libraries

## Kelly
(It's a secondary objective)
- [ ] make a separate module that implements
 - [ ] a Kelly "returns" object (it won't be called `AlphaStream` any more)
 - [ ] a Kelly constraint that does the risk part

That's pretty much it. we must also have a `**kwargs` into the policy object so we can pass solver options (max iters for SCS)
