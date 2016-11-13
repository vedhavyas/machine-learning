package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
)

func main() {

	k := flag.Int("K", 3, "Value K for the Benchmark")
	splitString := flag.String("Split", "0.6", "Spilit percentage of data. Should be between 0 - 1 float")
	flag.Parse()

	split, err := strconv.ParseFloat(*splitString, 32)
	if err != nil {
		log.Fatal(err)
	}

	model := &KNNModel{
		K:        *k,
		Split:    float32(split),
		FileName: "data/iris.data",
	}

	response, err := Benchmark(model)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(response)
}
