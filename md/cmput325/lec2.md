# CMPUT 325

## Lecture 2: Fun language

- List elements are seperated by a space
- In general, f( (x1 ... xn) ) --> x, notice the spacing for parameters
- `first( L )` returns first element of L, error if L is not a list or an empty list
- `rest( L )` returns L without the first element in L, error if L is not a list or an empty list
- Using compositions of calls to `first` (or `f`) and `rest` (or `r`), we can get any component (atom or sublist) from a given list, regardless of its depth
- For brevity, `f(f(r(L)))` can be written as `ffr(L)`
- `cons( x, L )` returns a new list K = (x y1 y2 y3 ... yn) where x is an atom and L = (y1 y2 y3 ... yn)
- In theory, there is a way to define any primitive function from scratch.
- Assume we have the following primitive functions:
  - Arithmetic: +, -, *, /, and so on
  - Comparators
  - `if`, `then`, `else`
  - `null(x)`
    - `true` iff `x` is an empty list
  - `atom(x)`
    - `true` iff `x` is an atom
  - `eq(x, y)`
    - `true` iff `x` and `y` are the same atom
    - `eq( a,a )` returns true
    - `eq( a,b )` returns false
    - `eq( (a), (a) )` returns false because the arguments are lists
- Similar idea is used to create RISC syntax.
- Remarks:
  - No notion of variable-as-storage
  - No assignment statement as in procedural languages
  - No loop constructs
  - **Recursion** as the only mechanism to define non-trivial functions
- 1-step evaluation = replacement + substitution
- Examples covered:
  `length( L )`, `append( L1, L2 )`, `last( L )`, `removeLast( L )`
- It makes sense to break the solution into smaller functions
- One function must do only one thing