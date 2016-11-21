package knn

import . "github.com/vedhavyas/machine-learning/base"

// KNNModel adds a few more attributes apart from existing BaseModel
// Required model to KNN algorithm
type KNNModel struct {
	BaseModel
	trainingSet [][]float64
	testingSet  [][]float64
	K           int
	Split       float32
}

// neighbour is a neighbour for a test instance
type neighbour struct {
	Distance float64
	Class    string
}

// neighbours is a plural class that implements sort interface
type neighbours []neighbour

// Len returns the size of neighbours
func (r neighbours) Len() int {
	return len(r)
}

// Less returns bool for i < j
func (r neighbours) Less(i, j int) bool {
	return r[i].Distance <= r[j].Distance
}

// Swap swaps the i and j neighbours
func (r neighbours) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}
