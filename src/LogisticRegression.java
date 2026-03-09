public class LogisticRegression {
    private final int numClasses;
    private final int numFeatures;
    private final double[][] weights;
    private final double[] bias;
    private final double learningRate;

    public LogisticRegression(int numClasses, int numFeatures, double learningRate) {
        this.numClasses = numClasses;
        this.numFeatures = numFeatures;
        this.learningRate = learningRate;
        this.weights = new double[numClasses][numFeatures];
        this.bias = new double[numClasses];
    }

    private double[] logits(double[] x) {
        double[] z = new double[numClasses];
        for (int c = 0; c < numClasses; c++) {
            double sum = bias[c];
            for (int j = 0; j < numFeatures; j++) {
                sum += weights[c][j] * x[j];
            }
            z[c] = sum;
        }
        return z;
    }

    private double[] softmax(double[] z) {
        double max = z[0];
        for (int i = 1; i < z.length; i++) {
            if (z[i] > max) max = z[i];
        }

        double sum = 0.0;
        double[] probs = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            probs[i] = Math.exp(z[i] - max);
            sum += probs[i];
        }

        for (int i = 0; i < z.length; i++) {
            probs[i] /= sum;
        }
        return probs;
    }

    public double[] predictProba(double[] x) {
        return softmax(logits(x));
    }

    public int predict(double[] x) {
        double[] probs = predictProba(x);
        int bestClass = 0;
        double bestProb = probs[0];

        for (int c = 1; c < numClasses; c++) {
            if (probs[c] > bestProb) {
                bestProb = probs[c];
                bestClass = c;
            }
        }
        return bestClass;
    }

    public void trainSample(double[] x, int label) {
        double[] probs = predictProba(x);

        for (int c = 0; c < numClasses; c++) {
            double y = (c == label) ? 1.0 : 0.0;
            double error = probs[c] - y; // gradient for cross-entropy + softmax

            for (int j = 0; j < numFeatures; j++) {
                weights[c][j] -= learningRate * error * x[j];
            }
            bias[c] -= learningRate * error;
        }
    }

    public void train(double[][] X, int[] y, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < X.length; i++) {
                trainSample(X[i], y[i]);
            }
        }
    }

    public double accuracy(double[][] X, int[] y) {
        int correct = 0;
        for (int i = 0; i < X.length; i++) {
            if (predict(X[i]) == y[i]) correct++;
        }
        return (double) correct / X.length;
    }
}