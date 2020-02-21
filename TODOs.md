# 2020

## Drop pd.Panel
- [ ] Change design of predictions. Panel is used for covariances and MPOReturnForecasts. Require from the user a function that returns
the prediction(s) (made at time tau) for time t.

## Drop support of example ipynb's 
- [ ] Move the non-explanatory code into cvxportfolio proper (e.g., factor model risk estimator).
- [ ] Move the rest into the documentation.

--- 
# 2017

## Examples
- [ ] modify data files so we don't store NaN (look at how you read them)
- [x] real time optimization:
 - [x] Excel output
 - [ ] FIX output
 - [x] rounding result trade vector (return rounded vector from the policy object).

## design
- [ ] impose that last column of returns is cash return. user shouldnt specify cash_key

## infrastructure
- [ ] makefile: support make docs and make pip
- [ ] clean up the gitignore for new stuff

## Tests
- [x] remove the pickle file, replace it with .csv or something
- [x] fix them
- [x] make them work with Travis
- [x] ideally, they should work with python2

## Misc
- [ ] make sure gammas are >= 0
- [ ] sector constraint should be equality
- [ ] cost -> constraint should be done, like costs=[gamma_leverage*Leverage()], and constraints=[Leverage()<= 3]
- [ ] documentation should be like cvxpy. we have variables constants parameters etc. we have a library of functions that manipulate basic objects, each one is documented... we have standard operators +- etc.

## Python 2
- [x] not sure which features of python3 (other than print() and __matmul__) we're using. it might be easy to support also python2.

## Kelly
(It's a secondary objective)
- [ ] make a separate module that implements
 - [ ] a Kelly "returns" object
 - [ ] a Kelly constraint that does the risk part
