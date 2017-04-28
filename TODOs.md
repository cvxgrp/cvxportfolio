## Tests
- [ ] fix them
- [ ] make them work with Travis
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
