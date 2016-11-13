package main

import "fmt"

type KNNModel struct {
	Split                  float32
	OriginalSet            [][]string
	TrainingSet            [][]string
	TestingSet             [][]string
	FileName               string
	K                      int
	ClassIndex             int
	AttributeIndexRange    int
	AttributeNormalisation bool
}

type BenchmarkResponse struct {
	K        int
	Accuracy float32
}

func (res BenchmarkResponse) String() string {
	return fmt.Sprintf(
		"K: %v\nAccuracy: %v",
		res.K,
		res.Accuracy*float32(100),
	)
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
