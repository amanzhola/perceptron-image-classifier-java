import java.awt.Color;

public class ImageClassifier {

    public static double[] extractFeatures(Picture picture) {
        int width = picture.width();
        int height = picture.height();

        double[] features = new double[width * height];
        int k = 0;

        for (int x = 0; x < width; x++) {
            for (int y = 0; y < height; y++) {
                Color color = picture.get(x, y);

                // Вариант 1: только красный канал
                // features[k++] = color.getRed() / 255.0;

                // Вариант 2: средняя яркость (лучше для большинства случаев)
                double gray = (color.getRed() + color.getGreen() + color.getBlue()) / 3.0 / 255.0;
                features[k++] = gray;
            }
        }

        return features;
    }

    public static void main(String[] args) {
        if (args.length < 2) {
            System.out.println("Usage:");
            System.out.println("java ImageClassifier <train.txt> <test.txt> [perceptron|logreg|mlp]");
            return;
        }

        String trainFile = args[0];
        String testFile = args[1];
        String modelType = (args.length >= 3) ? args[2].toLowerCase() : "perceptron";

        In in = new In(trainFile);
        In in1 = new In(testFile);

        int m = in.readInt();
        int width = in.readInt();
        int height = in.readInt();

        int trainCount = 0;
        while (!in.isEmpty()) {
            in.readString();
            in.readInt();
            trainCount++;
        }

        In trainIn = new In(trainFile);
        trainIn.readInt();
        trainIn.readInt();
        trainIn.readInt();

        double[][] trainFeatures = new double[trainCount][];
        int[] trainLabels = new int[trainCount];

        for (int i = 0; i < trainCount; i++) {
            String filename = trainIn.readString();
            Picture picture = new Picture(filename);
            trainFeatures[i] = extractFeatures(picture);
            trainLabels[i] = trainIn.readInt();
        }

        Object model = null;

        if (modelType.equals("perceptron")) {
            MultiPerceptron multiPerceptron = new MultiPerceptron(m, width * height);

            int epochs = 5;
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < trainCount; i++) {
                    multiPerceptron.trainMulti(trainFeatures[i], trainLabels[i]);
                }
            }

            model = multiPerceptron;
            System.out.println("Model: MultiPerceptron");
        }
        else if (modelType.equals("logreg")) {
            LogisticRegression lr = new LogisticRegression(m, width * height, 0.05);
            lr.train(trainFeatures, trainLabels, 10);
            model = lr;
            System.out.println("Model: Logistic Regression");
        }
        else if (modelType.equals("mlp")) {
            MLP mlp = new MLP(width * height, 64, m, 0.01);
            mlp.train(trainFeatures, trainLabels, 10);
            model = mlp;
            System.out.println("Model: MLP");
        }
        else {
            System.out.println("Unknown model: " + modelType);
            System.out.println("Allowed: perceptron, logreg, mlp");
            return;
        }

        in1.readInt();
        in1.readInt();
        in1.readInt();

        int errorCount = 0;
        int totalCount = 0;

        while (!in1.isEmpty()) {
            totalCount++;

            String filename = in1.readString();
            Picture picture = new Picture(filename);
            double[] features = extractFeatures(picture);

            int label = in1.readInt();
            int predictedLabel;

            if (model instanceof MultiPerceptron) {
                predictedLabel = ((MultiPerceptron) model).predictMulti(features);
            }
            else if (model instanceof LogisticRegression) {
                predictedLabel = ((LogisticRegression) model).predict(features);
            }
            else {
                predictedLabel = ((MLP) model).predict(features);
            }

            if (predictedLabel != label) {
                errorCount++;
            }
        }

        double errorRate = (double) errorCount / totalCount;
        System.out.println("test error rate = " + errorRate);
    }
}
