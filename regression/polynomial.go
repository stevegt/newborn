package regression

import (
	"math"
	"math/rand"
	"time"
	// . "github.com/stevegt/goadapt"
)

type Polynomial struct {
	Bias         float64
	Coefficients [][]float64
}

func New() *Polynomial {
	return &Polynomial{}
}

// The model starts to learn
func (p *Polynomial) Train(features [][]float64, values []float64, degree int, learningRate float64, steps int, lambda, tolerance float64) (cost float64, i int) {
	if len(features) != len(values) {
		panic("number of data and number of values should be the same")
	}

	// length of features[0] shows the number of variables
	p.setRandomVariables(degree, len(features[0]))

	prevCost := math.MaxFloat64
	for i = 0; i < steps; i++ {
		cost, _, _ = p.gradientDescent(features, values, learningRate, lambda)
		dCost := math.Abs(cost - prevCost)
		if dCost < tolerance {
			break
		}
		prevCost = cost
	}
	return
}

// Cost computes the cost and derivatives of the cost function with
// respect to the model parameters (coefficients and bias).
//
// In gradient descent, we want to minimize the cost function.  The
// cost function is defined as:
//
//	1/2m * sum((h(x) - y)^2) + lambda * sum(c^2)
//
// ...where m is the number of training examples, h(x) is the
// predicted value, y is the actual value, lambda is the
// regularization parameter, and c is a coefficient.  The first term
// is the mean squared error and the second term is the regularization
// term.  The regularization term is used to prevent overfitting.
//
// Using the chain rule, we can compute the derivative of the cost
// function with respect to the bias and coefficients.  The chain
// rule states that the derivative of a function f(g(x)) is:
//
//	f'(g(x)) * g'(x)
//
// The derivative of the cost function with respect to the bias is:
//
//	2 * 1/2m * sum((h(x) - y)^2)
//
//	1/m * sum(h(x) - y)
//
// The derivative of the cost function with respect to the
// coefficients is:
//
//	1/m * sum((h(x) - y) * x).
func (p *Polynomial) Cost(features [][]float64, values []float64, lambda float64) (cost float64, d_bias float64, derivatives [][]float64) {
	m := len(features)

	// Compute predictions, errors, and the sum of the squared errors (cost).
	predictions := make([]float64, m)
	errors := make([]float64, m)
	for i := 0; i < m; i++ {
		fs := features[i]
		predicted := p.Predict(fs)
		predictions[i] = predicted
		errors[i] = predicted - values[i]
		cost += math.Pow(errors[i], 2)
	}

	// Compute the derivative of the bias cost function.
	sigma := float64(0)
	for i := 0; i < m; i++ {
		// The derivative of x^2 is 2x, which can be cancelled out by
		// multiplying the learning rate by 1/2.
		sigma += errors[i]
	}
	d_bias = float64(1) / float64(m) * sigma

	// Compute the derivatives of the coefficient costs.
	derivatives = make([][]float64, len(p.Coefficients))
	// k is the degree of the coefficient.
	for k, _ := range derivatives {
		d := make([]float64, len(p.Coefficients[0]))
		// j is the feature index.
		for j := range d {
			sigma := float64(0)
			// i is the index of the training example.
			for i := 0; i < m; i++ {
				// calculate the derivative of the cost function with
				// respect to the coefficient.
				feature := features[i][j]
				sigma += errors[i] * math.Pow(feature, float64(k+1))
			}
			// Lambda is a regularization parameter in machine
			// learning and statistics which, when added, can help to
			// avoid the risk of overfitting. Lambda is typically
			// applied in Ridge regression or Lasso regression, which
			// are techniques to add a penalty to the size of
			// coefficients in order to prevent overfitting.
			//
			// In the following equation, `lambda` is controlling the
			// amount of shrinkage. The coefficients of the features
			// are being reduced, or shrunk, and this helps to reduce
			// their importance in the model. The larger the lambda,
			// the more the coefficients are penalized, thereby
			// helping to reduce overfitting.
			d[j] = float64(1) / float64(m) * (sigma + lambda*p.Coefficients[k][j])
		}
		derivatives[k] = d
	}

	return
}

func (p *Polynomial) Predict(features []float64) float64 {
	v := p.Bias

	for i, c := range p.Coefficients {
		for j, f := range features {
			v += c[j] * math.Pow(f, float64(i+1))
		}
	}

	// Assert(!math.IsNaN(v), Spf("v is NaN, Bias: %f, Coefficients: %f", p.Bias, p.Coefficients))
	// Assert(!math.IsInf(v, 0), Spf("v is Inf, Bias: %f, Coefficients: %f", p.Bias, p.Coefficients))

	return v
}

func (p *Polynomial) setRandomVariables(degree int, numOfVariables int) {
	rand.Seed(time.Now().UnixNano())
	bias := rand.Float64()

	coefficients := make([][]float64, degree)
	for i := range coefficients {
		c := make([]float64, numOfVariables)
		for j := range c {
			c[j] = rand.Float64()
		}
		coefficients[i] = c
	}

	p.Bias = bias
	p.Coefficients = coefficients
}

func (p *Polynomial) gradientDescent(features [][]float64, values []float64, learningRate float64, lambda float64) (cost float64, d_bias float64, d_coefficients [][]float64) {
	cost, d_bias, d_coefficients = p.Cost(features, values, lambda)
	p.Bias -= learningRate * d_bias
	if math.IsNaN(p.Bias) || math.IsInf(p.Bias, 0) {
		p.Bias = rand.Float64()
	}

	for m, c := range p.Coefficients {
		for n := range c {
			p.Coefficients[m][n] -= learningRate * d_coefficients[m][n]
			if math.IsNaN(p.Coefficients[m][n]) || math.IsInf(p.Coefficients[m][n], 0) {
				p.Coefficients[m][n] = rand.Float64()
			}
			// Assert(!math.IsNaN(p.Coefficients[m][n]), Spf("p.Coefficients[%d][%d] is NaN, d_bias: %f, d_coefficients: %f", m, n, d_bias, d_coefficients[m][n]))
			// Assert(!math.IsInf(p.Coefficients[m][n], 0), Spf("p.Coefficients[%d][%d] is Inf, d_bias: %f, d_coefficients: %f", m, n, d_bias, d_coefficients[m][n]))
		}
	}
	return
}
