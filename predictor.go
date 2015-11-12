// Package collaborativepermute implements an online collaborative permutation
// learner.
// 
// In many machine learning problems, we seek to learn how much various people
// prefer a varied set of objects. Using the accelerated Trace norm prediction
// algorithm from (Wang KDD '14), we can both generate directed queries and
// form predictions based on those queries. As a result, the number of questions
// required is dramatically reduced.
//
// Example
//
// Suppose we have three people and five movies, and we wish to recover which
// movies are preferred by each person. Suppose also that we can only ask
// five questions of these people, but that we decide which person to ask next.
//
// To do that, run the following:
//
// 	eng := collaborativepermute.NewEngine(3, 5)
// 	
// 	for i := 0; i < 5; i++ {
// 		q := eng.Generate(-1)
// 		// display q to user, update order of q.Choices
// 		q.Respond(q)
// 	}
//
// Currently, the implementation will only ever ask about two items at a time.
// If you cannot decide when each user is prompted (such as for an online form),
// pass the current user's ID to .Generate to restrict the queries generated.
package collaborativepermute

import (
	"github.com/fatlotus/gauss"
	"math"
	"fmt"
	"math/rand"
)

// Struct predictor implements a basic learning engine.
type Engine struct {
	X, Xp, Z gauss.Array
	Nu, Alpha, Lambda, T float64
	History []Query
}

// Struct Query represents a prompt to the user.
type Query struct {
	User int
	Choices []int
	weight float64
}

// NewEngine allocates and initializes a learning engine for the given corpus
// size. By default, users consider all elements equally.
func NewEngine(users, choices int) *Engine {
	return &Engine{
		X: gauss.Zero(users, choices),
		Xp: gauss.Zero(users, choices),
		Z: gauss.Zero(users, choices),
		History: make([]Query, 0),
		Nu: 1,
		Lambda: 0.04,
		Alpha: 1,
		T: 1,
	}
}

func (p *Engine) hingeLoss(samps []Query) float64 {
	sum := 0.0
	for _, x := range samps {
		diff := *p.X.I(x.User, x.Choices[0]) - *p.X.I(x.User, x.Choices[1])
		sum += math.Max(1 - diff, 0)
	}
	return sum / float64(len(samps))
}

func (p *Engine) gradientLoss(samps []Query) gauss.Array {
	result := gauss.Zero(p.X.Shape...)
	before := p.hingeLoss(samps)
	for i := range result.Data {
		p.X.Data[i] += 0.0001
		result.Data[i] = (p.hingeLoss(samps) - before) / 0.0001
		p.X.Data[i] -= 0.0001
	}

	return result
}

func (p *Engine) update(samps []Query) {
	alphaP := (1 + math.Sqrt(1 + 4*p.Alpha*p.Alpha)) / 2

	U, S, V := gauss.SVD(gauss.Sum(p.Z, gauss.Scale(
		p.gradientLoss(samps), -p.Nu)))
	for i := range S.Data {
		S.Data[i] = math.Max(0, S.Data[i] - p.Lambda)
	}

	p.Xp = p.X
	p.X = gauss.Product(gauss.Product(U, gauss.Diagonal(S.Data)), V.Transpose())
	p.Z = gauss.Sum(p.X, 
		gauss.Scale(
			gauss.Sum(p.X, gauss.Scale(p.Xp, -1)), ((p.Alpha - 1) / (alphaP))))
	p.Alpha = alphaP
}

// Method Respond takes a completed Prompt and updates the engine's 
// belief matrix.
func (p *Engine) Respond(prompt Query) error {
	if len(prompt.Choices) != 2{
		return fmt.Errorf("can only handle binary rankings")
	}
	if prompt.User < 0 || prompt.User >= p.X.Shape[0] {
		return fmt.Errorf("must have 0 <= user [%d] < %d",
			prompt.User, p.X.Shape[0])
	}
	for _, choice := range prompt.Choices {
		if choice < 0 || choice >= p.X.Shape[1] {
			return fmt.Errorf("must have 0 <= choice [%d] < %d",
				choice, p.X.Shape[1])
		}
	}
	p.History = append(p.History, prompt)
	p.update(p.History)
	return nil
}

// Function Generate creates a new Query to display to the user.
//
// If user is non-negative, only return queries for that user. Otherwise, return
// the query that would be the most helpful.
func (p *Engine) Generate(user int) Query {
	candidates := make([]Query, 0)
	sum := 0.0
	for u := 0; u < p.X.Shape[0]; u++ {
		if user >= 0 && user != u {
			continue
		}

		for a := 0; a < p.X.Shape[1]; a++ {
			for b := 0; b < p.X.Shape[1]; b++ {
				if a == b {
					continue
				}

				diff := math.Abs(*p.X.I(u, a) - *p.X.I(u, b))
				weight := math.Exp(-diff / p.T)
				sum += weight
				candidates = append(candidates, Query{
					User: u,
					Choices: []int{ a, b },
					weight: weight,
				})
			}
		}
	}
	
	offset := rand.Float64() * sum
	for _, option := range candidates {
		if offset < option.weight {
			if *p.X.I(option.User, option.Choices[0]) <
			   *p.X.I(option.User, option.Choices[1]) {
				option.Choices[0], option.Choices[1] = option.Choices[1], option.Choices[0]
			}
			return option
		}
		offset -= option.weight
	}
	
	panic("Could not find another question")
}