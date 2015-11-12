# Active Collaborative Permutation Learning

In many machine learning problems, we seek to learn how much various people
prefer a varied set of objects. Using the accelerated Trace norm prediction
algorithm from ([Wang KDD '14][1]), we can both generate directed queries and
form predictions based on those queries. As a result, the number of questions
required is dramatically reduced.

[1]: http://ttic.uchicago.edu/~nati/Publications/WangSrebroEvans2014.pdf

## Example

Suppose we have three people and five movies, and we wish to recover which
movies are preferred by each person. Suppose also that we can only ask
five questions of these people, but that we decide which person to ask next.

To do that, run the following:

```go
eng := collaborativepermute.NewEngine(3, 5)

for i := 0; i < 5; i++ {
	q := eng.Generate(-1)
	// display q to user, update order of q.Choices
	q.Respond(q)
}
```

If you cannot decide when each user is prompted (such as for an online form),
pass the current user's ID to `.Generate` to restrict the queries generated.

## License

The code in this repository is covered under the MIT License:

> Copyright (c) 2015 Jeremy Archer
> 
> Permission is hereby granted, free of charge, to any person obtaining
> a copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
> 
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.
> 
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
> EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
> MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
> NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
> BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
> ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
> CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
