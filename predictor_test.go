package collaborativepermute

import (
	"math/rand"
	"testing"
	"fmt"
)

// In this example we are allowing the predictor to select which user to query.
func ExampleEngine_all() {
	rand.Seed(23)
	eng := NewEngine(2, 2)

	for i := 0; i < 3; i++ {
		q := eng.Generate(-1)
		fmt.Printf("user %v: %v?\n", q.User, q.Choices)
		q.Choices = []int{0, 1}
		eng.Respond(q)
	}
	// Output:
	// user 1: [1 0]?
	// user 0: [1 0]?
	// user 0: [0 1]?
}

// In this example, we decide ahead of time which user will receive the
// question.
func ExampleEngine_one() {
	rand.Seed(23)
	eng := NewEngine(2, 2)

	for i := 0; i < 3; i++ {
		q := eng.Generate(0)
		fmt.Printf("user %v: %v?\n", q.User, q.Choices)
		q.Choices = []int{0, 1}
		eng.Respond(q)
	}
	// Output:
	// user 0: [1 0]?
	// user 0: [0 1]?
	// user 0: [0 1]?
}

func TestConvergence(t *testing.T) {
	rand.Seed(23)
	eng := NewEngine(10, 10)
	incorrect := 0

	for i := 0; i < 300; i++ {
		q := eng.Generate(-1)
		if q.Choices[0] == q.Choices[1] {
			t.Fatalf("asked to compare %d with itself", q.Choices[0])
		}
		if q.Choices[0] >= q.Choices[1] {
			q.Choices[0], q.Choices[1] = q.Choices[1], q.Choices[0]
			incorrect += 1
		}
		eng.Respond(q)
	}
	
	if incorrect > 40 {
		t.Fatalf("needed %v mistakes for a 10x10 matrix", incorrect)
	}
}