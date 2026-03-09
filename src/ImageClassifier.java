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
                features[k++] = color.getRed() / 255.0;
            }
        }

        return features;
    }

    public static void main(String[] args) {
        In in = new In(args[0]);
        In in1 = new In(args[1]);

        int m = in.readInt();
        int width = in.readInt();
        int height = in.readInt();

        int trainCount = 0;
        while (!in.isEmpty()) {
            in.readString();
            in.readInt();
            trainCount++;
        }

        In trainIn = new In(args[0]);
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

        MultiPerceptron multiPerceptron = new MultiPerceptron(m, width * height);

        int epochs = 5;
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < trainCount; i++) {
                multiPerceptron.trainMulti(trainFeatures[i], trainLabels[i]);
            }
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
            int predictedLabel = multiPerceptron.predictMulti(features);

            if (predictedLabel != label) {
                errorCount++;
            }
        }

        double errorRate = (double) errorCount / totalCount;
        System.out.println("test error rate = " + errorRate);
    }
}
