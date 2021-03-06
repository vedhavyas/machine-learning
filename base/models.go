package base

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// BaseModel contains basic attributes and methods to get going
type BaseModel struct {
	FileName               string
	ClassIndex             int
	Data                   [][]float64
	categoricalAttributes  map[int]map[string]float64
	AttributeNormalisation bool
}

// GetClassString returns the category of the given normalised value
// returns zero string if not found
func (model *BaseModel) GetClassString(value float64) string {
	categories := model.categoricalAttributes[model.ClassIndex]
	for k, v := range categories {
		if v == value {
			return k
		}
	}

	return ""
}

//setCategoricalAttributeIndex will initiate the given index as category
func (model *BaseModel) setCategoricalAttributeIndex(index int) {
	if model.categoricalAttributes == nil {
		model.categoricalAttributes = make(map[int]map[string]float64)
	}

	_, ok := model.categoricalAttributes[index]
	if ok {
		return
	}

	model.categoricalAttributes[index] = make(map[string]float64)
}

// checkForCategoricalAttributes will check if the any attribute in the given data is category
// do that we can normalise the data if needed
func (model *BaseModel) checkForCategoricalAttributes(instances [][]string) {
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

// updateClassIndex will update the class index if default
func (model *BaseModel) updateClassIndex(data []string) {
	if model.ClassIndex == -1 {
		model.ClassIndex = len(data) - 1
	}

	fmt.Printf("class index is set at %v\n", model.ClassIndex)
}

// assignCategoryValues will assign a value for each category for a given attribute
func (model *BaseModel) assignCategoryValues(instances [][]string) {
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

// normaliseData normalise the data set that is loaded
func (model *BaseModel) normaliseData(instances [][]string) {
	// update class index
	model.updateClassIndex(instances[0])

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
		model.Data = normaliseSet
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

	model.Data = normaliseSet

}

// PrepareModel will load the data from the file given and
// Normalise the data if required
func (model *BaseModel) PrepareModel() error {
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
