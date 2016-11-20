package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

type PerceptronModel struct {
	FileName               string
	KFold                  int
	LearningRate           float64
	Epochs                 int
	data                   [][]float64
	ClassIndex             int
	categoricalAttributes  map[int]map[string]float64
	AttributeNormalisation bool
}

func (model *PerceptronModel) getClassString(value float64) string {
	categories := model.categoricalAttributes[model.ClassIndex]
	for k, v := range categories {
		if v == value {
			return k
		}
	}

	return ""
}

func (model *PerceptronModel) setCategoricalAttributeIndex(index int) {
	if model.categoricalAttributes == nil {
		model.categoricalAttributes = make(map[int]map[string]float64)
	}

	_, ok := model.categoricalAttributes[index]
	if ok {
		return
	}

	model.categoricalAttributes[index] = make(map[string]float64)
}

func (model *PerceptronModel) checkForCategoricalAttributes(instances [][]string) {
	// we try to convert the attributes in the first row to float64
	// any failures is considered categorical
	// normalisation is must if one category is found
	// we will not categorise class for obvious reasons
	instance := instances[0]
	indexes := len(instance)
	for i := 0; i < indexes; i++ {
		if i == model.ClassIndex {
			model.setCategoricalAttributeIndex(i)
			continue
		}
		_, err := strconv.ParseFloat(instance[i], 64)
		if err == nil {
			continue
		}

		// must be a categorical value
		model.setCategoricalAttributeIndex(i)
		model.AttributeNormalisation = true
	}
}

func (model *PerceptronModel) assignCategoryValues(instances [][]string) {
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

func (model *PerceptronModel) normaliseData(instances [][]string) {
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
		model.data = normaliseSet
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

	model.data = normaliseSet

}

func (model *PerceptronModel) loadData() error {
	fmt.Printf("loading data from '%v'...\n", model.FileName)
	fh, err := os.Open(model.FileName)
	if err != nil {
		return err
	}

	rows, err := csv.NewReader(fh).ReadAll()
	if err != nil {
		return err
	}

	var originalSet [][]string
	for _, row := range rows {
		for i := range row {
			row[i] = strings.TrimSpace(row[i])
		}
		originalSet = append(originalSet, row)
	}
	fmt.Printf("loaded %v rows...\n", len(originalSet))

	fmt.Println("normalising data...")
	model.normaliseData(originalSet)
	fmt.Println("normalisation done...")

	return nil
}
