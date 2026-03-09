import java.util.Random;

public class MLP {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private final double learningRate;

    private final double[][] W1; // hiddenSize x inputSize
    private final double[] b1;
    private final double[][] W2; // outputSize x hiddenSize
    private final double[] b2;

    private final Random rnd = new Random(42);

    public MLP(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        W1 = new double[hiddenSize][inputSize];
        b1 = new double[hiddenSize];
        W2 = new double[outputSize][hiddenSize];
        b2 = new double[outputSize];

        initWeights();
    }

    private void initWeights() {
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] = (rnd.nextDouble() - 0.5) * 0.01;
            }
        }

        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] = (rnd.nextDouble() - 0.5) * 0.01;
            }
        }
    }

    private double relu(double x) {
        return Math.max(0.0, x);
    }

    private double reluDerivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
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

    public int predict(double[] x) {
        double[] probs = predictProba(x);
        int best = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[best]) {
                best = i;
            }
        }
        return best;
    }

    public double[] predictProba(double[] x) {
        double[] z1 = new double[hiddenSize];
        double[] a1 = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            double sum = b1[i];
            for (int j = 0; j < inputSize; j++) {
                sum += W1[i][j] * x[j];
            }
            z1[i] = sum;
            a1[i] = relu(sum);
        }

        double[] z2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = b2[i];
            for (int j = 0; j < hiddenSize; j++) {
                sum += W2[i][j] * a1[j];
            }
            z2[i] = sum;
        }

        return softmax(z2);
    }

    public void trainSample(double[] x, int label) {
        double[] z1 = new double[hiddenSize];
        double[] a1 = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            double sum = b1[i];
            for (int j = 0; j < inputSize; j++) {
                sum += W1[i][j] * x[j];
            }
            z1[i] = sum;
            a1[i] = relu(sum);
        }

        double[] z2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double sum = b2[i];
            for (int j = 0; j < hiddenSize; j++) {
                sum += W2[i][j] * a1[j];
            }
            z2[i] = sum;
        }

        double[] probs = softmax(z2);

        // output gradient
        double[] dz2 = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            double y = (i == label) ? 1.0 : 0.0;
            dz2[i] = probs[i] - y;
        }

        // gradients for W2 and b2
        double[][] dW2 = new double[outputSize][hiddenSize];
        double[] db2 = new double[outputSize];

        for (int i = 0; i < outputSize; i++) {
            db2[i] = dz2[i];
            for (int j = 0; j < hiddenSize; j++) {
                dW2[i][j] = dz2[i] * a1[j];
            }
        }

        // hidden gradient
        double[] da1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            double sum = 0.0;
            for (int i = 0; i < outputSize; i++) {
                sum += W2[i][j] * dz2[i];
            }
            da1[j] = sum;
        }

        double[] dz1 = new double[hiddenSize];
        for (int j = 0; j < hiddenSize; j++) {
            dz1[j] = da1[j] * reluDerivative(z1[j]);
        }

        // gradients for W1 and b1
        double[][] dW1 = new double[hiddenSize][inputSize];
        double[] db1 = new double[hiddenSize];

        for (int i = 0; i < hiddenSize; i++) {
            db1[i] = dz1[i];
            for (int j = 0; j < inputSize; j++) {
                dW1[i][j] = dz1[i] * x[j];
            }
        }

        // update W2, b2
        for (int i = 0; i < outputSize; i++) {
            b2[i] -= learningRate * db2[i];
            for (int j = 0; j < hiddenSize; j++) {
                W2[i][j] -= learningRate * dW2[i][j];
            }
        }

        // update W1, b1
        for (int i = 0; i < hiddenSize; i++) {
            b1[i] -= learningRate * db1[i];
            for (int j = 0; j < inputSize; j++) {
                W1[i][j] -= learningRate * dW1[i][j];
            }
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