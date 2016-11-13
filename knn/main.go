package main

import (
	"flag"
	"fmt"
	"log"
)

func main() {

	k := flag.Int("k", 3, "Value K for the Benchmark")
	split := flag.Float64("split", 0.6, "Spilit percentage of data. Should be between 0 - 1 float")
	flag.Parse()

	if *split > 1.0 {
		log.Fatal("Split cannot be grater than 1.0")
	}

	model := &KNNModel{
		K:                      *k,
		Split:                  float32(*split),
		FileName:               "data/iris.data",
		AttributeIndexRange:    4,
		ClassIndex:             4,
		AttributeNormalisation: false,
	}

	response, err := Benchmark(model)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response)
}
