package main

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/stevegt/newborn/regression"
)

func main() {
	//headers, content := data.ReadCSVData("./data/dataset_test.csv")
	//fmt.Println(headers)
	//
	//x := make([]float64, len(content[headers[0]]))
	//y := make([]float64, len(content[headers[1]]))
	//
	//for i := 0; i < len(content[headers[0]]); i++ {
	//	x[i], _ = strconv.ParseFloat(content[headers[0]][i], 64)
	//	y[i], _ = strconv.ParseFloat(content[headers[1]][i], 64)
	//}
	//
	//data.ScatterPlot(x, y, headers[0], headers[1], "example")

	//headers, content := data.ReadCSVData("./regression/dataset_test.csv")
	//features := make([][]float64, len(content[headers[0]]))
	//for i := range features {
	//	f := make([]float64, len(headers)-1)
	//	for j := range f {
	//		f[j], _ = strconv.ParseFloat(content[headers[j]][i], 64)
	//	}
	//	features[i] = f
	//}
	//
	//values := make([]float64, len(content[headers[0]]))
	//for i := range values {
	//	values[i],_ = strconv.ParseFloat(content[headers[len(headers) - 1]][i], 64)
	//}

	//p := &regression.Polynomial{
	//	Bias: 2,
	//	Coefficients: [][]float64{
	//		{3, 2, 1},
	//		{7, 1, 8},
	//		{4, 6, 2}},
	//}

	// matrix of {x1, x2, y} where y = 3*x1 + 2*x2^2
	matrix := [][]float64{
		{1, 4, 35},
		{2, 6, 78},
		{3, 8, 137},
		{4, 2, 20},
		{5, 9, 177},
		{6, 1, 20},
		{7, 4, 53},
		{8, 4, 56},
		{9, 3, 45},
		{10, 6, 102},
		{11, 7, 131},
		{12, 0, 36},
	}

	// split matrix into features and values
	features := make([][]float64, len(matrix))
	values := make([]float64, len(matrix))
	for i, row := range matrix {
		features[i] = row[:len(row)-1]
		values[i] = row[len(row)-1]
	}

	p := regression.New()
	p.Train(features, values, 2, 0.00028, 4000, 0)
	fmt.Println(p.Bias)
	fmt.Println(p.Coefficients)

	// add a few more features and values to test interpolation,
	// but don't train on these
	for i := 0; i < 10; i++ {
		x1 := rand.Float64()*10 + 1
		x2 := rand.Float64()*10 + 1
		y := 3*x1 + 2*x2*x2
		features = append(features, []float64{x1, x2})
		values = append(values, y)
	}

	for i, f := range features {
		// if we don't have a value for this feature, use NaN
		var v float64
		if i < len(values) {
			v = values[i]
		} else {
			v = math.NaN()
		}
		// show the predicted value for these features
		result := p.Predict(f)
		fmt.Println(f, v, result)
	}

	/*
		headers, content := data.ReadCSVData("./knn/dataset_test.csv")
		currentData := make([][]float64, len(content[headers[0]]))
		for i := range currentData {
			f := make([]float64, len(headers)-1)
			for j := range f {
				f[j], _ = strconv.ParseFloat(content[headers[j]][i], 64)
			}
			currentData[i] = f
		}

		labels := make([]int, len(content[headers[0]]))
		for i := range labels {
			labels[i], _ = strconv.Atoi(content[headers[len(headers)-1]][i])
		}

		newData := []float64{57.0, 1.0, 4.0, 140.0, 192.0, 0.0, 0.0, 148.0, 0.0, 0.4, 2.0, 0.0, 6.0}

		label := knn.KNN(currentData, labels, newData, 3)
		fmt.Println(label)
	*/
}
