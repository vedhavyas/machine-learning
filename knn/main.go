package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"strings"
)

func main() {
	log.SetFlags(log.Lshortfile)

	k := flag.Int("k", 3, "Value K for the Benchmark")
	split := flag.Float64("split", 0.6, "Spilit percentage of data. Should be between 0 - 1 float")
	fileName := flag.String("data", "", "Data File")
	attributeRange := flag.String("attr-range", "", "Attribute range. Min:Max. Both inclusive")
	classIndex := flag.String("class-index", "", "Index of the class in a given row")
	flag.Parse()

	if *split > 1.0 {
		log.Fatal("Split cannot be grater than 1.0")
	}

	if *fileName == "" {
		log.Fatal("Data file cannot be empty")
	}

	if *attributeRange == "" {
		log.Fatal("Attribute range cannot be empty")
	}
	attrs := strings.Split(*attributeRange, ":")
	attrStart, err := strconv.Atoi(attrs[0])
	if err != nil {
		log.Fatal(err)
	}

	attrEnd, err := strconv.Atoi(attrs[1])
	if err != nil {
		log.Fatal(err)
	}

	if *classIndex == "" {
		log.Fatal("Class Index cannot be empty")
	}

	classIndexInt, err := strconv.Atoi(*classIndex)
	if err != nil {
		log.Fatal(err)
	}

	model := &KNNModel{
		K:                      *k,
		Split:                  float32(*split),
		FileName:               *fileName,
		AttributeIndexStart:    attrStart,
		AttributeIndexEnd:      attrEnd,
		ClassIndex:             classIndexInt,
		AttributeNormalisation: false,
	}

	response, err := Benchmark(model)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response)
}
