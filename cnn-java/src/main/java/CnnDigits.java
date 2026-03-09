import java.io.File;
import java.util.Random;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class CnnDigits {

    public static void main(String[] args) throws Exception {
        String trainPath = "..\\datasets\\digits\\training";
        String testPath  = "..\\datasets\\digits\\testing";

        int height = 28;
        int width = 28;
        int channels = 1;
        int numClasses = 10;
        int batchSize = 64;
        int epochs = 2;
        int seed = 123;

        if (args.length >= 2) {
            trainPath = args[0];
            testPath = args[1];
        }

        File trainDir = new File(trainPath);
        File testDir = new File(testPath);

        if (!trainDir.exists() || !testDir.exists()) {
            System.out.println("Train/test folder not found.");
            System.out.println("Train: " + trainDir.getAbsolutePath());
            System.out.println("Test : " + testDir.getAbsolutePath());
            return;
        }

        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

        FileSplit trainSplit = new FileSplit(trainDir, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));
        FileSplit testSplit = new FileSplit(testDir, NativeImageLoader.ALLOWED_FORMATS, new Random(seed));

        System.out.println("Train images: " + trainSplit.length());
        System.out.println("Test images: " + testSplit.length());

        ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
        trainRR.initialize(trainSplit);

        ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
        testRR.initialize(testSplit);

        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, numClasses);
        DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, numClasses);

        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(trainIter);
        trainIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(0.001))
                .l2(1e-4)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .nIn(channels)
                        .nOut(16)
                        .stride(1, 1)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new ConvolutionLayer.Builder(3, 3)
                        .nOut(32)
                        .stride(1, 1)
                        .convolutionMode(ConvolutionMode.Same)
                        .activation(Activation.RELU)
                        .build())
                .layer(new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nOut(128)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numClasses)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        for (int epoch = 1; epoch <= epochs; epoch++) {
            System.out.println("Epoch " + epoch + " / " + epochs);
            model.fit(trainIter);
            trainIter.reset();

            Evaluation eval = model.evaluate(testIter);
            System.out.println(eval.stats());

            double accuracy = eval.accuracy();
            double errorRate = 1.0 - accuracy;
            System.out.println("test error rate = " + errorRate);

            testIter.reset();
        }
    }
}