package knn

import . "github.com/vedhavyas/machine-learning/base"

type KNNModel struct {
	BaseModel
	trainingSet [][]float64
	testingSet  [][]float64
	K           int
	Split       float32
}

type Neighbour struct {
	Distance float64
	Class    string
}

type Neighbours []Neighbour

func (r Neighbours) Len() int {
	return len(r)
}

func (r Neighbours) Less(i, j int) bool {
	return r[i].Distance <= r[j].Distance
}

func (r Neighbours) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}
