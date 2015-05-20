package classifier;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.estimators.KernelEstimator;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Created by IntelliJ IDEA.
 * User: Batu
 * Date: 2009.12.30.
 * Time: 17:11:59
 * To change this template use File | Settings | File Templates.
 */
public class PatternClassifier extends Classifier {

    private int length;
    private final Map<Object, List<KernelEstimator>> KernelEstimators = new HashMap<Object, List<KernelEstimator>>();
    private Object[] classes;
    private List<KernelEstimator> positiveEstimators;
    private List<KernelEstimator> negativeEstimators;
    private static final double PRECISION = 0.01;
    private static final double WEIGHT = 1;
    public static final int POSITIVE_CLASS = 1;
    public static final int NEGATIVE_CLASS = 0;


    private class EstimatorInitializerThread implements Runnable {

        private final Object classValue;

        private EstimatorInitializerThread(Object classValue) {
            this.classValue = classValue;
        }

        public void run() {
            List<KernelEstimator> ke = new Vector<KernelEstimator>(length);
            for (int i = 0; i < length; ++i) {
                ke.add(new KernelEstimator(PRECISION));
            }
            KernelEstimators.put(classValue, ke);

        }
    }

    private class TrainingDataInitializerThread implements Runnable {

        private final Instance instance;

        private TrainingDataInitializerThread(Instance instance) {
            this.instance = instance;
        }

        public void run() {
            if (instance == null) {
                throw new IllegalArgumentException("Null instance found.");
            }
            double classValue = instance.classValue();
            List<KernelEstimator> estimators = KernelEstimators.get(classes[(int) classValue]);
            for (int i = 0; i < length; ++i) {
                estimators.get(i).addValue(instance.value(i), WEIGHT);
            }
        }
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        if (instances == null) {
            throw new IllegalArgumentException("No instances provided for training.");
        }

        length = instances.numAttributes();

        Attribute classAttribute = instances.classAttribute();
        Enumeration classValues = classAttribute.enumerateValues();
        ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        while (classValues.hasMoreElements()) {
            es.submit(new EstimatorInitializerThread(classValues.nextElement())).get();
        }
        while (es.isTerminated()) {
            es.awaitTermination(100, TimeUnit.MILLISECONDS);
        }
        es.shutdown();
        classes = KernelEstimators.keySet().toArray();

        es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
        Enumeration<Instance> instEnum = instances.enumerateInstances();

        while (instEnum.hasMoreElements()) {
            /*Instance instance = instEnum.nextElement();
            if (instance == null) {
                throw new IllegalArgumentException("Null instance found.");
            }
            double classValue = instance.classValue();
            List<KernelEstimator> estimators = KernelEstimators.get(classes[(int) classValue]);
            for (int i = 0; i < length; ++i) {
                estimators.get(i).addValue(instance.value(i), WEIGHT);
            }*/
            es.submit(new TrainingDataInitializerThread(instEnum.nextElement())).get();
        }
        while (es.isTerminated()) {
            es.awaitTermination(100, TimeUnit.MILLISECONDS);
        }
        es.shutdown();
        //System.out.println("KernelEstimators = " + KernelEstimators);
    }

    @Override
    public double classifyInstance(Instance is) {

        if (is == null) {
            throw new IllegalArgumentException("Null instance cannot be classified");
        }


        double[] votes = new double[classes.length];

        for (int i = 0; i < length - 1; ++i) {
            double value = is.value(i);
            double maxProbability = KernelEstimators.get(classes[0]).get(i).getProbability(value);
            int maxIndex = 0;
            for (int j = 1; j < classes.length; ++j) {
                double probability = KernelEstimators.get(classes[j]).get(i).getProbability(value);
                if (probability > maxProbability) {
                    maxProbability = probability;
                    maxIndex = j;
                }
            }
            votes[maxIndex]++;

        }
        double maxClassValue = votes[0];
        int maxClassIndex = 0;
        for (int j = 1; j < classes.length; ++j) {
            if (votes[j] > maxClassValue) {
                maxClassValue = votes[j];
                maxClassIndex = j;
            }
        }
        //System.out.println(Arrays.toString(votes));
        if (is.classAttribute().isNumeric()) {
            return (Double) classes[maxClassIndex];
        }

        return maxClassIndex;
    }

    @Override
    public double[] distributionForInstance(Instance is) throws Exception {

        if (is == null) {
            throw new IllegalArgumentException("Null instance cannot be classified");
        }


        double[] votes = new double[classes.length];

        for (int i = 0; i < length - 1; ++i) {
            double value = is.value(i);
            double maxProbability = KernelEstimators.get(classes[0]).get(i).getProbability(value);
            int maxIndex = 0;
            for (int j = 1; j < classes.length; ++j) {
                double probability = KernelEstimators.get(classes[j]).get(i).getProbability(value);
                if (probability > maxProbability) {
                    maxProbability = probability;
                    maxIndex = j;
                }
            }
            votes[maxIndex]++;
        }

        double total = 0;

        for (int i = 0; i < classes.length; ++i) {
            total += votes[i];

        }

        for (int i = 0; i < classes.length; ++i) {
            votes[i] /= total;
        }

        return votes;
    }


}
