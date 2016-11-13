package main

import (
	"fmt"
	"strconv"
)

type KNNModel struct {
	originalSet            [][]string
	trainingSet            [][]string
	testingSet             [][]string
	FileName               string
	K                      int
	ClassIndex             int
	AttributeIndexRange    int
	categoricalAttributes  []int
	Split                  float32
	AttributeNormalisation bool
}

func (model *KNNModel) IsCategoricalAttribute(index int) bool {
	for _, v := range model.categoricalAttributes {
		if v == index {
			return true
		}
	}

	return false
}

func (model *KNNModel) SetCategoricalAttribute(index int) {
	if model.IsCategoricalAttribute(index) {
		return
	}
	model.categoricalAttributes = append(model.categoricalAttributes, index)
	model.AttributeNormalisation = true
}

func (model *KNNModel) NormaliseData() {

}

func (model *KNNModel) CheckForCategoricalAttributes() {
	// we try to convert the attributes in the first row to float64
	// any failures is considered categorical
	instance := model.originalSet[0]
	for i := 0; i < model.AttributeIndexRange; i++ {
		_, err := strconv.ParseFloat(instance[i], 64)
		if err == nil {
			continue
		}

		// must be a categorical values
		model.SetCategoricalAttribute(i)
	}
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
