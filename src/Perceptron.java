public class Perceptron {

    private int n;
    private double[] weights;

    public Perceptron(int n) {
        this.n = n;
        this.weights = new double[n];
    }

    public int numberOfInputs() {
        return n;
    }

    public double weightedSum(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += weights[i] * x[i];
        }
        return sum;
    }

    public int predict(double[] x) {
        if (weightedSum(x) > 0) {
            return +1;
        }
        return -1;
    }

    public void train(double[] x, int label) {
        if (predict(x) != label) {
            for (int i = 0; i < n; i++) {
                weights[i] += label * x[i];
            }
        }
    }

    public String toString() {
        String temp = "(";
        for (int i = 0; i < weights.length; i++) {
            temp += weights[i];
            if (i < weights.length - 1) {
                temp += ", ";
            }
        }
        temp += ")";
        return temp;
    }

    public static void main(String[] args) {
        double[] training1 = { 5.0, -4.0, 3.0 };
        double[] training2 = { 2.0, 3.0, -2.0 };
        double[] training3 = { 4.0, 3.0, 2.0 };
        double[] training4 = { -6.0, -5.0, 7.0 };

        Perceptron perceptron = new Perceptron(3);
        StdOut.println(perceptron);
        perceptron.train(training1, +1);
        StdOut.println(perceptron);
        perceptron.train(training2, -1);
        StdOut.println(perceptron);
        perceptron.train(training3, +1);
        StdOut.println(perceptron);
        perceptron.train(training4, -1);
        StdOut.println(perceptron);
    }
}
