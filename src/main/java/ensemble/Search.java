package ensemble;

import org.apache.commons.math.stat.descriptive.DescriptiveStatistics;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

import java.util.*;


public enum Search {

    ALL,
    SINGLE_BEST,
    FORWARD_SEARCH,
    BACKWARD_SEARCH;

    private List<Classifier> classifiers;
    private Instances dataset;
    private int numFolds;
    private FusionType fusionType;

    public void setClassifiers(List<Classifier> classifiers) {
        this.classifiers = classifiers;
    }

    public void setDataset(Instances dataset) {
        this.dataset = dataset;
    }

    public void setNumFolds(int numFolds) {
        this.numFolds = numFolds;
    }

    public void setFusionType(FusionType fusionType) {
        this.fusionType = fusionType;
    }

    private List<Classifier> all() {
        return classifiers;
    }

    
    /**
     * ?
     * @param classifier
     * @return
     */
    private double evaluateSingle(Classifier classifier) {
        DescriptiveStatistics accStat = new DescriptiveStatistics();
        dataset.randomize(new Random());
        for (int i = 0; i < numFolds; ++i) {
            Instances train = dataset.trainCV(numFolds, i);
            Instances test = dataset.testCV(numFolds, i);
            try {
                classifier.buildClassifier(train);
                Enumeration<Instance> instEnum = test.enumerateInstances();
                double tp = 0;
                double fp = 0;
                double tn = 0;
                double fn = 0;
                while (instEnum.hasMoreElements()) {
                    Instance is = instEnum.nextElement();
                    double classValue = is.classValue();

                    if (classValue != classifier.classifyInstance(is)) {
                        if (classValue == 0) {
                            fn++;
                        } else {
                            fp++;
                        }
                    } else {
                        if (classValue == 1) {
                            tp++;
                        } else {
                            tn++;
                        }
                    }
                }
                double acc = (tp + tn) / (tp + fn + tn + fp);
                if (Double.isNaN(acc)) {
                    acc = 0;
                }
                accStat.addValue(acc);
            } catch (Exception e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }

        }
        return accStat.getMean();
    }

    
    /**
     * ?
     * @param array
     */
    private void normalize(double[] array) {
        double sum = 0;
        for (int i = 0; i < array.length; ++i) {
            sum += array[i];
        }
        for (int i = 0; i < array.length; ++i) {
            array[i] /= sum;
            if (Double.isNaN(array[i])) {
                array[i] = 1.0 / (double)array.length;
            }
        }
    }

/**
 * Beállítja a súlyt?
 * @param ensemble
 * @return
 */
    public double[] assignWeights(List<Classifier> ensemble) {
        double[] weights = new double[ensemble.size()];
        for (int i = 0; i < weights.length; ++i) {
            double accuracy = evaluateSingle(ensemble.get(i));
            weights[i] = Math.log(accuracy / (1.0 - accuracy));
            if (Double.isNaN(weights[i])) {
                weights[i] = 1.0 / (double)weights.length;
            }
        }
        normalize(weights);
        System.out.println(Arrays.toString(weights));
        return weights;
    }

/**
 * ?
 * @param ensemble
 * @return
 */
    private double evaluateEnsemble(List<Classifier> ensemble) {
        int size = ensemble.size();
        boolean isWeighted = fusionType == FusionType.WEIGHTED_MAJORITY;
        double[] weights;
        if (isWeighted) {
            weights = assignWeights(ensemble);
            fusionType.setWeights(weights);
        }

       
        DescriptiveStatistics accStat = new DescriptiveStatistics();
        dataset.randomize(new Random());
        for (int i = 0; i < numFolds; ++i) {
            Instances train = dataset.trainCV(numFolds, i);
            Instances test = dataset.testCV(numFolds, i);
            try {
                for (Classifier classifier : ensemble) {
                    classifier.buildClassifier(train);

                }
                Enumeration<Instance> instEnum = test.enumerateInstances();
                double tp = 0;
                double fp = 0;
                double tn = 0;
                double fn = 0;
                while (instEnum.hasMoreElements()) {
                    Instance is = instEnum.nextElement();
                    double classValue = is.classValue();
                    double[] votes = new double[size];
                    for (int j = 0; j < size; ++j) {
                        if (fusionType == FusionType.MAJORITY || fusionType == FusionType.WEIGHTED_MAJORITY) {
                            votes[j] = classifiers.get(j).classifyInstance(is);
                        } else {
                            votes[j] = classifiers.get(j).distributionForInstance(is)[1];
                        }
                    }
                    fusionType.setVotes(votes);
                    double prediction = fusionType.apply();
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
                double acc = (tp + tn) / (tp + fn + tn + fp);
                if (Double.isNaN(acc)) {
                    acc = 0;
                }
                accStat.addValue(acc);
            } catch (Exception e) {
                e.printStackTrace();  //To change body of catch statement use File | Settings | File Templates.
            }

        }
        return accStat.getMean();
    }

    /**
     * Visszaada a legjobb osztályozókból álló listát
     * @param list
     * @return A legjobb osztályozó lista
     */
    private Classifier selectBest(List<Classifier> list) {
        Classifier best = list.get(0);
        double bestScore = evaluateSingle(best);
        int size = list.size();
        for (int i = 1; i < size; ++i) {
            Classifier current = list.get(i);
            double currentScore = evaluateSingle(current);
            if (currentScore > bestScore) {
                best = current;
                bestScore = currentScore;
            }
        }
        return best;
    }

    /**
     * Visszaadja a legrosszabb osztályozókból álló listát
     * @param list 
     * @return A legrosszabb osztályozó lista
     */
    private Classifier selectWorst(List<Classifier> list) {
        Classifier worst = list.get(0);
        double worstScore = evaluateSingle(worst);
        int size = list.size();
        for (int i = 1; i < size; ++i) {
            Classifier current = list.get(i);
            double currentScore = evaluateSingle(current);
            if (currentScore < worstScore) {
                worst = current;
                worstScore = currentScore;
            }
        }
        return worst;
    }

    /**
     * 
     * @param list
     * @return
     */
    private Classifier selectRandom(List<Classifier> list) {
        Random random = new Random();
        return list.get(random.nextInt(list.size()));
    }

    /**
     * A legjobb osztályozóhoz hozzárendeli egyenként a többi osztályozót egészen addig amíg meg nem találja az optimális eredményt
     * @return Optimális keresési eredmény
     */
    private List<Classifier> forwardSearch() {
        List<Classifier> optimal = new ArrayList<Classifier>();
        List<Classifier> avaliable = new ArrayList<Classifier>(classifiers);
        List<Classifier> temp = new ArrayList<Classifier>();
        Classifier best = selectBest(avaliable);
        avaliable.remove(best);
        optimal.add(best);
        double bestScore = evaluateSingle(best);
        while (avaliable.size() > 0) {
            System.out.println("bestScore = " + bestScore);
            System.out.println("avaliable = " + avaliable.size());
            best = selectRandom(avaliable);
            System.out.println("best = " + best.getClass().getName());
            optimal.add(best);
            double currentScore = evaluateEnsemble(optimal);
            System.out.println("currentScore = " + currentScore);
            if (currentScore > bestScore) {
                avaliable.remove(best);
                bestScore = currentScore;
                avaliable.addAll(temp);
                temp.clear();
            } else {
                optimal.remove(best);
                avaliable.remove(best);
                temp.add(best);
            }
        }
        return optimal;
    }

    /**
     * Az osztályozó listából folyamatosan kivesz 1-1 osztályozót, hogy összállítsa a legoptimálisabb osztályozó listát
     * @param classifierList Osztályozó lista
     * @return Optimális keresési eredmény
     */
    
    private List<Classifier> backwardSearch(List<Classifier> classifierList) {

        List<Classifier> optimal = new ArrayList<Classifier>(classifierList);
        double bestScore = evaluateEnsemble(optimal);
        System.out.println("bestScore = " + bestScore);
        int size = optimal.size();
        int index = -1;
        for (int i = 0; i < size && optimal.size() > 1; ++i) {
            List<Classifier> temp = new ArrayList<Classifier>(classifierList);
            temp.remove(i);
            double currentScore = evaluateEnsemble(temp);
            System.out.println("currentScore = " + currentScore);
            if (currentScore > bestScore) {
                bestScore = currentScore;
                index = i;
            }
        }
        if (index == -1 || optimal.size() == 1) {
            return optimal;
        } else {
            optimal.remove(index);
            return backwardSearch(optimal);
        }
    }
/**
 * Visszaadja a keresés tipusától függõen az optimális osztályozó listát
 * @return List<Classifier> az optmális osztályozó lista
 */
    public List<Classifier> select() {
        if (classifiers == null || classifiers.isEmpty() || dataset == null || numFolds < 1 || fusionType == null) {
            throw new IllegalArgumentException();
        }
        switch (this) {
            case ALL:
                return all();
            case FORWARD_SEARCH:
                return forwardSearch();
            case BACKWARD_SEARCH:
                return backwardSearch(classifiers);
            case SINGLE_BEST:
                List<Classifier> list = new ArrayList<Classifier>();
                Classifier best = selectBest(classifiers);
                System.out.println("best.getClass().getName() = " + best.getClass().getName());
                list.add(best);
                return list;
            default:
                throw new UnsupportedOperationException();
        }
    }
}
