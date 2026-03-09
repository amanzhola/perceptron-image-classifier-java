public class MultiPerceptron {

    private int m;
    private int n;
    private Perceptron[] perceptrons;

    public MultiPerceptron(int m, int n) {
        this.m = m;
        this.n = n;
        this.perceptrons = new Perceptron[m];

        for (int i = 0; i < m; i++) {
            perceptrons[i] = new Perceptron(n);
        }
    }

    public int numberOfClasses() {
        return m;
    }

    public int numberOfInputs() {
        return n;
    }

    public int predictMulti(double[] x) {
        int predictedLabel = 0;
        double max = perceptrons[0].weightedSum(x);

        for (int i = 1; i < m; i++) {
            double sum = perceptrons[i].weightedSum(x);
            if (sum > max) {
                max = sum;
                predictedLabel = i;
            }
        }

        return predictedLabel;
    }

    public void trainMulti(double[] x, int label) {
        int predicted = predictMulti(x);

        if (predicted != label) {
            perceptrons[predicted].train(x, -1);
            perceptrons[label].train(x, +1);
        }
    }

    public String toString() {
        String temp = "(";
        for (int i = 0; i < m; i++) {
            temp += perceptrons[i].toString();
            if (i < m - 1) {
                temp += ", ";
            }
        }
        temp += ")";
        return temp;
    }

    public static void main(String[] args) {
        int m = 2;
        int n = 3;

        double[] training1 = { 5.0, -4.0, 3.0 };
        double[] training2 = { 2.0, 3.0, -2.0 };
        double[] training3 = { 4.0, 3.0, 2.0 };
        double[] training4 = { -6.0, -5.0, 7.0 };

        MultiPerceptron perceptron = new MultiPerceptron(m, n);
        StdOut.println(perceptron);
        perceptron.trainMulti(training1, 1);
        StdOut.println(perceptron);
        perceptron.trainMulti(training2, 0);
        StdOut.println(perceptron);
        perceptron.trainMulti(training3, 1);
        StdOut.println(perceptron);
        perceptron.trainMulti(training4, 0);
        StdOut.println(perceptron);
    }
}
