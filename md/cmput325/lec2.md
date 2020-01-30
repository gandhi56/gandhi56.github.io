# CMPUT 325

## Lecture 2: Fun language
### Fun
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

## Lecture 2: Intro to LISP
* `reverse( L )` using `append( L1, L2 )`
* Abstract data type for binary tree
  * Goal: implement a binary tree and some operations, such as inserting elements
  * Two main tasks:
    * Decide how trees are represented by lists
    * Implement an abstract data type for binary trees and the operations on them
  * User will work with trees using only these functions. The user is protected from the details of our data representation
  * Bottom up construction
* Tree representation
  * Empty tree: `nil`
  * Nonempty tree: `(left-subtree, node-value, right-subtree)`
  * Selectors:
    * `leftTree( Tr ) = f( Tr )`
    * `rightTree( Tr ) = f( r( r( Tr ) ) )`
    * `nodeValue( Tr ) = f( r( Tr ) )`
  * Constructors:
    * `consNilTr() = nil`
      * return an empty tree
    * `consTree(L, V, R) = cons( L, cons( V, cons( R, nil ) ) )`
      * construct tree with subtrees L, R and value V
  * Test: 
    * `isEmpty( Tr ) = eq( Tr, nil )`
    * return true iff `Tr` is an empty tree
* Building an abstract data type
  * Functions are the only ones that need direct knowledge of our tree representation
  * Everything else can be implemented in terms of these basic functions - providing such a base set of functions is the essence of implementing an abstract data type in functional programming

* `insert` into tree
  * assume our trees contain integer values and are sorted such that every value in the left subtree < node value < all values in right subtree
  * unique values
  * `insert( Tr, int )` inserts `int` into the binary tree `Tr`

  ![1](1.png)

### LISP

* interpreted language
* case insensitive
* uses read-eval-print-loop (REPL) similar to a shell such as bash
  - read input
  - evaluate input
  - print result of evaluation
  - loop back to beginning
* Functions are defined by `(defun function-name parameter-list body)`
  * Example: 
    - Function definition: `(defun plus (x y) (+ x y))`
    - Function application: `(plus 3 4)`
* Lisp always interprets `(e1 e2 e3 ...)` as a function application. Use quote to "atomify" the expression
* An empty list is represented by either `()` or `nil`. Both are considered the same atom in Lisp.
* `nil` also represents false
* `T` represents true

## Lecture 4
* `(if condition then-part else-part)` is a special function because not every block is run, unlike other functions
* `trace` to see calls and returns to specific functions
* `untrace` stops the tracing
* functions can take variable number of arguments
* `(let ((x 3) (y 4)) (* (+ x y) x))` evaluates expression, but replaces names `x` and `y` with their values 3 and 4
* `let` does not allow using one variable to define another, use `let*` instead
* `eq` is true iff both are equal _atoms_, runs in a single machine instruction
* `equal` is more general
* ```
  (cond (P1 S1)
        (P2 S2)
        (P3 S3)
        ...
        (T Sn)
  )
  ```
* General form of `cond` (do not use it):
  ```
  (cond (P1 S11 S12 ... S1m)
  (P2 S21 S22 ... S2m)
  ...
  (T Sn1 Sn2 ... Snm)
  )
  ```
  * If `P1` is true then evaluate `S11`, `S12`, ... and `S1m` and return the result of evaluating `S1m`

* `list`
* `caar`, `cddadr`, etc.
* `print` and `format` for printing, strings
* `random`
* Use `quote` when everything is constant
* Use `list` when some contents are the result of evaluating functions
* Use `cons` for the result in recursion when you have computed a first element and the rest of a list
* Use `cons` for dotted pairs
* Use `cons` when your professor tells you to in a test
* `(car (cdr (car (cdr (cdr (cdr L)))))) = (cadr (cadddr L))`
  * max 4 levels deep
* Simple printing `(print arg)`
* Formatted printing: `(format t format-string arg1 ...)`
* `(random N)` generates a uniformly random integer from 0..N-1 if N is an integer
* `(random F)` generates a uniformly random floating point number in range [0..F)
* Accumulators
  * helper function with an extra parameter
  * the extra parameter accumulates the required result
  * Issues with simple recursion:
    * no real computation until hits the base case
    * all computation happens on return from recursion
  * Example 1: `reverse` using an accumulator
    ```
    (defun reverse_helper (L ResultSoFar)
      (if (null L)
        ResultSoFar
        (reverse_helper (cdr L)
          (cons (car L) ResultSoFar))
        )
      )

    (defun reverseAC (L)
      (reverse_helper L nil)
      )
    ```
  * Example 2: accumulating two types of results
    ![2](2.png)
  * Comparing accumulator with standard recursion
    * Standard recursion on a list
      * recurse to the end of list
      * compute result on return from recursion
      * bottom-up computation
    * Accumulators
      * accumulates results-so-far
      * computes results top-down
      * needs an extra accumulator variable for partial result
    * Questions to think about to decide whether accumulators should be used
      * top-down or bottom-up?
* Programming loops in LISP
  * In pure functional programming, we use recursion instead of loops although LISP has loop constructs (`for`, `do`, `loop`, ...)
  * break loop into two steps:
    * what to do in each run through the loop
    * how to solve the rest of the problem by recursion

## Lecture 5
* Symbolic expressions (S-Expressions, s-expr, sexpr)
  * universal data structure for Lisp
  * generalization of atoms and lists
    * all atoms and lists are sexpr
    * but not all atom and lists are sexpr
  * "dotted pair" `(x . y)`
  * Definition
    * atom is an s-expression
    * if `x1`, `x2`, ..., `xn` are s-expressions then (`x1` ... `xn`) is an s-expression
    * if `x1` and `x2` are s-expressions, then `( x1 . x2 )` is an s-expression (a dotted pair)
    * Examples:
      * `hello`
      * `(a b c)`
      * `(a (b) (()))`
      * `(a . b)`
      * `(a . (b . c))`
      * `(1 2 3 (4 . 5))`
  * `(car (x . y))` returns x
  * `(cdr (x . y))` returns y
  * `(car (cons 'x 'y)) = x`
  * `(cdr (cons 'x 'y)) = y`
  * `.` must be surrounded by whitespace: `(a.b)` is a list containing the atom `a.b`
  * machine-level representations
    * Example 1: `(cons 1 2)` or `(1 . 2)`
      ![3](3.png)

    * Example 2: `( (1 . (2 . 3)) . 4 )`
      ![4](4.png)
    
    * Example 3: List representation `(1 2 3 4)`
      ![5](5.png)

  * `(a . nil) = (a)` because `(cdr '(a . nil)) = (cdr ('(a))) = nil`
  * Every list can be written as nested dotted pairs:
    `(1 . (2 . (3 . (4 . nil))))`
  * Why use dotted pairs?
    * saves memory
    * simplifies direct access

## Lecture 6
* Higher order functions
  * Definition: a function that takes other function(s) as input and/or produce function(s) as output
  * often used to seperate:
    * a computation pattern
    * specific repeated action
  * Example 1:
    * Pattern: iterate over a list
    * Action: Compute the same function for each list element
  * Example 2:
    * Pattern: reduce a list to a single result
    * Action: reduce two arguments to one
  * Some typical higher order functions
    * Map - apply some function to all elements of a list
    * Reduce - apply two argument function repeatedly
    * Filter - select list elements that pass a test
    * Vector - apply many functions to one element
  * Why define high order functions?
    * use case: a common computation pattern, where the details can vary
    * removing code duplication
    * `apply` and `funcall` tells Lisp that there is a function to be called
      * only differs in syntax, same functionality
      * `(apply function-name (arg1 ... argn))`
      * `(funcall function-name arg1 ... argn)`

## Lecture 7
* Lambda Functions
  * get rid of named functions, why?
    * a function was the result of a higher order function
    * tried to return this newly computed function
  * lambda functions are function definitions without names
    * Syntax: `(lambda (x1 ... xn) body)`
    * Example: `((lambda (x y) (+ x y)) 5 3)`
* Lisp-1 and Lisp-2
  * Lisp-1 systems: "values" and functions in the same namespace
  * Lisp-2 systems: in seperate spaces
  * Common Lisp standard requires Lisp-2
  * if we have a variable bound to a function ...
  * ... we need to tell SBCL this is a function to be called
  * Consequence:
    * Working with lambda functions is much messier in Lisp-2 systems than in Lisp-1
* `function`
  * takes lambda function as its argument
  * returns function definition in an internal format used by SBCL
  * compiles it and returns an internal representation of the compiled code
  * representation is called **closure**
  * next, the function in the closure can be called in an application
  * use `funcall` or `apply` for the application
* Lambda Calculus
  * formal, abstract language
  * all functions are defined without giving them names
  * Lisp is based on lambda calculus but adds a large language on top of it
  * formal language with only four concepts:
    ```
    [identifier]  := a | b | ...
    [function]    := (lambda (x) [expression])
    [application] := ([expression] [expression])
    [expression]  := [identifier] | [application] | [function]
    ```
    * `identifier` corresponds to atom in Lisp
    * `function` is a lambda function in Lisp
    * `[expression]` can be an arbitrary lambda expression. It plays the role of the body in the function definition
    * In practice, this means: either the body is written directly or the body is computed by evaluating a lambda application
* Unary vs N-ary functions
  `[function] := (lambda (x) [expression])`
  * we only have unary functions - functions that take one parameter
  * any n-ary function (function with n arguments) can be defined using a series of unary functions
  * consequence:
    * to understand the model of computation for general functional programming
    * it is enough to understand computation with unary functions
* Curried functions
  * Goal: define an n-ary functions by a series of unary functions
  * can solve this by using higher order functions
  * main idea:
    * series of n unary function applications
    * each application processes one argument
    * the application produces a new function which has this argument hardcoded
  * Intuition
    * Example: `(plus 5 2)` is a function with two args
    * `(plus5 2)` is a function with one argument, the "add 5" is hardcoded into the new function `plus5`
    * function takes only the first argument
    * produces as result a new function
    * this function now takes the second argument
    * it produces a result a new function
    * etc.
    * the function that takes the last argument will have other argument values "hardcoded"
    * each function is computed on the fly by all the previous function applications

## Lecture 8

### Reductions in Lambda Calculus
* Goal: reduce a lambda expression to its **simplest possible form**
* This process is called *operational semantics* of lambda calculus
* In lambda calculus, computation is the process of reductions from one expression to another expression
* Example:
  ```
  ((lambda (x) (x 2)) (lambda (z) (+ z 1))) → (+ 2 1)
  ```
* Shorthand notation in lambda calculus
  ($\lambda$x ( + x 1)) for 
  ```
  (lambda (x) (+ x 1))
  ```
* In lambda calculus we do not need any of the primitive functions
* numbers can be represented by lambda expressions
* Questions about reductions
  * what type of reductions are there?
  * how do we do them?
  * is there a simplest form for a given expression?
  * is there always a simplest form?
  * is it unique?
  * how can we compute it?
  * can we compute it efficiently?
* Beta reduction
  * most intuitive and important one is what we called function application
  * called beta-reduction in the theory
  * we write $\rightarrow^{\beta}$ to indicate such a reduction
  * rule: