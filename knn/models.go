package main

import (
	"fmt"
	"sort"
	"strconv"
)

type KNNModel struct {
	normaliseSet           [][]float64
	trainingSet            [][]float64
	testingSet             [][]float64
	FileName               string
	K                      int
	ClassIndex             int
	AttributeIndexStart    int
	AttributeIndexEnd      int
	categoricalAttributes  map[int]map[string]float64
	Split                  float32
	AttributeNormalisation bool
}

func (model *KNNModel) GetClassString(value float64) string {
	categories := model.categoricalAttributes[model.ClassIndex]
	for k, v := range categories {
		if v == value {
			return k
		}
	}

	return ""
}

func (model *KNNModel) SetCategoricalAttributeIndex(index int) {
	if model.categoricalAttributes == nil {
		model.categoricalAttributes = make(map[int]map[string]float64)
	}

	_, ok := model.categoricalAttributes[index]
	if ok {
		return
	}

	model.categoricalAttributes[index] = make(map[string]float64)
}

func (model *KNNModel) checkForCategoricalAttributes(instances [][]string) {
	// we try to convert the attributes in the first row to float64
	// any failures is considered categorical
	// normalisation is must if one category is found
	// we will not categorise class for obvious reasons
	instance := instances[0]
	indexes := len(instance)
	for i := 0; i < indexes; i++ {
		if i == model.ClassIndex {
			model.SetCategoricalAttributeIndex(i)
			continue
		}
		_, err := strconv.ParseFloat(instance[i], 64)
		if err == nil {
			continue
		}

		// must be a categorical value
		model.SetCategoricalAttributeIndex(i)
		model.AttributeNormalisation = true
	}
}

func (model *KNNModel) assignCategoryValues(instances [][]string) {
	for index, categories := range model.categoricalAttributes {
		var value float64
		for _, instance := range instances {
			category := instance[index]
			_, ok := categories[category]
			if ok {
				continue
			}

			categories[category] = value
			value += 1
		}
	}
}

func (model *KNNModel) NormaliseData(instances [][]string) {
	// look for categories
	model.checkForCategoricalAttributes(instances)

	//convert all categorical values to respective floats
	model.assignCategoryValues(instances)

	// instantiate normalisedSet
	outerIndex := len(instances)
	innerIndex := len(instances[0])
	normaliseSet := make([][]float64, outerIndex)

	// convert to float64
	for index, instance := range instances {
		normaliseSet[index] = make([]float64, innerIndex)
		for i, k := range instance {
			categories, ok := model.categoricalAttributes[i]
			if ok {
				normaliseSet[index][i] = categories[k]
				continue
			}

			value, _ := strconv.ParseFloat(k, 64)
			normaliseSet[index][i] = value
		}
	}

	// do normalisation if required
	if !model.AttributeNormalisation {
		model.normaliseSet = normaliseSet
		return
	}

	for i := 0; i < innerIndex; i++ {
		if i == model.ClassIndex {
			continue
		}

		var column sort.Float64Slice
		for _, instance := range normaliseSet {
			column = append(column, instance[i])
		}

		sort.Sort(column)
		min := column[0]
		den := column[len(column)-1] - min

		for _, instance := range normaliseSet {
			num := instance[i] - min
			instance[i] = num / den
		}
	}

	model.normaliseSet = normaliseSet

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
