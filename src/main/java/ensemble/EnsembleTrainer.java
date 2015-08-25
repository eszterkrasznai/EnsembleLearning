package ensemble;


import classifier.PatternClassifier;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.ConsistencySubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.ADTree;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.stemmers.NullStemmer;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.GregorianCalendar;
import java.util.List;
import java.util.Random;

class EnsembleTrainer{

    private final List<Classifier> classifiers;
    private final FusionType fusionType;
    private final Search search;
    private final int numFolds;
    private List<Classifier> optimalSet;
    private Instances dataset;
    private DescriptiveStatistics senStat;
    private DescriptiveStatistics speStat;
    private DescriptiveStatistics accStat;
    private static List<NullValueStatistic> nullValueStatistic;
    
    private EnsembleTrainer(List<Classifier> classifiers, FusionType fusionType, Search search, int numFolds) {
        this.classifiers = classifiers;
        this.fusionType = fusionType;
        this.search = search;
        this.numFolds = numFolds;
    }

    void train(Instances training) throws Exception {
        search.setClassifiers(classifiers);
        search.setDataset(training);
        search.setNumFolds(numFolds);
        search.setFusionType(fusionType);
        optimalSet = search.select();
        for (Classifier classifier : optimalSet) {
                classifier.buildClassifier(training);
        }
        if (fusionType == FusionType.WEIGHTED_MAJORITY && optimalSet.size() != 1) {
            fusionType.setWeights(search.assignWeights(optimalSet));
        }
    }

    double test(Instance is) throws Exception {
        int size = optimalSet.size();
        double[] votes = new double[size];

        for (int j = 0; j < size; ++j) {
            if (fusionType == FusionType.MAJORITY || fusionType == FusionType.WEIGHTED_MAJORITY) {
                votes[j] = optimalSet.get(j).classifyInstance(is);
            } else {
                votes[j] = optimalSet.get(j).distributionForInstance(is)[1];
            }
        }
        fusionType.setVotes(votes);
        return fusionType.apply();
    }


    private void loadDataset(String path) throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(path));
        dataset = loader.getDataSet();
    }
    void validate(String datasetPath, int cvFolds, boolean filterAttributes) throws Exception {
        senStat = new DescriptiveStatistics();
        speStat = new DescriptiveStatistics();
        accStat = new DescriptiveStatistics();
        loadDataset(datasetPath);
        dataset.randomize(new Random());
        if (filterAttributes) {
            AttributeSelection as = new AttributeSelection();
                as.setEvaluator(new ConsistencySubsetEval());
                as.SelectAttributes(dataset);
                as.reduceDimensionality(dataset);
                ASEvaluation ase = new InfoGainAttributeEval();
                ase.buildEvaluator(dataset);
        }


        search.setClassifiers(classifiers);
        search.setDataset(dataset);
        search.setNumFolds(cvFolds);
        search.setFusionType(fusionType);
        List<Classifier> optimalSet = search.select();
        for (int i = 0; i < cvFolds; ++i) {
            Instances trainDataset = dataset.trainCV(cvFolds, i);
            train(trainDataset);
            Instances test = dataset.testCV(cvFolds, i);

                for (Classifier classifier : optimalSet) {
                    classifier.buildClassifier(trainDataset);
                }
                Enumeration<Instance> instEnum = test.enumerateInstances();
                double tp = 0;
                double fp = 0;
                double tn = 0;
                double fn = 0;
                search.setDataset(trainDataset);
                if (fusionType == FusionType.WEIGHTED_MAJORITY && optimalSet.size() != 1) {
                    fusionType.setWeights(search.assignWeights(optimalSet));
                }
                while (instEnum.hasMoreElements()) {
                    Instance is = instEnum.nextElement();
                    double classValue = is.classValue();
                    double prediction = test(is);
                    if (prediction > 0.5) {
                        prediction = 1.0;
                    } else {
                        prediction = 0.0;
                    }
                    if (classValue != prediction) {
                        if (classValue == 0.0) {
                            fn++;
                        } else {
                            fp++;
                        }
                    } else {
                        if (classValue == 1.0) {
                            tp++;
                        } else {
                            tn++;
                        }
                    }
                }

                double sen = tp / (tp + fn);
                if (!Double.isNaN(sen)) {
                    senStat.addValue(sen);
                }
                double spe = tn / (fp + tn);
                if (!Double.isNaN(spe)) {
                    speStat.addValue(spe);
                }
                double acc = (tp + tn) / (tp + tn + fp + fn);
                if (!Double.isNaN(acc)) {
                    accStat.addValue(acc);
                }
                
                nullValueStatistic.add(new NullValueStatistic(fn,acc,fusionType));
                
                System.out.println("nullas ertek: " + fn);
                System.out.println("egyes ertek: " + tp);
                System.out.println(i +". kiertekeles pontossaga:" + acc);
                System.out.println("FusionType: " + fusionType);
        }
    }

    public DescriptiveStatistics getSenStat() {
        return senStat;
    }

    public DescriptiveStatistics getSpeStat() {
        return speStat;
    }

    public DescriptiveStatistics getAccStat() {
        return accStat;
    }

    // This is the main method of the application (My first test commit)
    public static void main(String[] args) {

        List<Classifier> classifiers = new ArrayList<Classifier>();
        classifiers.add(new ADTree());
        classifiers.add(new IBk(33));
        classifiers.add(new AdaBoostM1());
        classifiers.add(new LibSVM());
        classifiers.add(new NaiveBayes());
        classifiers.add(new RandomTree());
        classifiers.add(new PatternClassifier());
        classifiers.add(new MultilayerPerceptron());
        
        nullValueStatistic = new ArrayList<NullValueStatistic>();
        
        Search s = Search.BACKWARD_SEARCH;
        for(FusionType fusi : FusionType.values()) {
        	 FusionType ft = fusi;
             System.out.println(s.name() + " " + ft.name());
             EnsembleTrainer trainer = new EnsembleTrainer(classifiers, ft, s, 10);
             try {
                 trainer.validate("C:/Users/Eszter/Desktop/test.arff",10,true);
                 System.out.println(trainer.getAccStat());
             } catch (Exception e) {
                 e.printStackTrace();
             }
            
        }
        
        Date now = new Date();
        System.out.println("toString(): " + now);
        
        for(NullValueStatistic n : nullValueStatistic)  {
        	n.toString();
        }
    }
}
