package main

import (
	"flag"
	"fmt"
	"log"
	"os"
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
	normalise := flag.Bool("norm", false, "True to force normalise the attributes")
	flag.Parse()

	if *split > 1.0 {
		log.Println("Split cannot be grater than 1.0")
		flag.Usage()
		os.Exit(1)
	}

	if *fileName == "" {
		log.Println("Data file cannot be empty")
		flag.Usage()
		os.Exit(1)
	}

	if *attributeRange == "" {
		log.Println("Attribute range cannot be empty")
		flag.Usage()
		os.Exit(1)
	}
	attrs := strings.Split(*attributeRange, ":")
	attrStart, err := strconv.Atoi(attrs[0])
	if err != nil {
		log.Println(err)
		flag.Usage()
		os.Exit(1)
	}

	attrEnd, err := strconv.Atoi(attrs[1])
	if err != nil {
		log.Println(err)
		flag.Usage()
		os.Exit(1)
	}

	if *classIndex == "" {
		log.Println("Class Index cannot be empty")
		flag.Usage()
		os.Exit(1)
	}

	classIndexInt, err := strconv.Atoi(*classIndex)
	if err != nil {
		log.Println(err)
		flag.Usage()
		os.Exit(1)
	}

	model := &KNNModel{
		K:                      *k,
		Split:                  float32(*split),
		FileName:               *fileName,
		AttributeIndexStart:    attrStart,
		AttributeIndexEnd:      attrEnd,
		ClassIndex:             classIndexInt,
		AttributeNormalisation: *normalise,
	}

	response, err := Benchmark(model)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response)
}
