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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
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

    /**
     * ?
     * @param training
     * @throws Exception
     */
    void train(Instances training) throws Exception {
    	// A keres�nek be�ll�tom az oszt�lyoz�kat
        search.setClassifiers(classifiers);
        // A keres�nek be�ll�tom a tanul�si adathalmaz�t
        search.setDataset(training);
        // A keres�nek megmondom, hogy h�ny r�szre van felosztva az adathalmaz?
        search.setNumFolds(numFolds);
        // A keres�nek be�ll�tom, hogy milyen fusiontype-al dolgozzon
        search.setFusionType(fusionType);
        
        optimalSet = search.select();
        for (Classifier classifier : optimalSet) {
        	//Itt t�rt�nik a tanul�s
                classifier.buildClassifier(training);
        }
        if (fusionType == FusionType.WEIGHTED_MAJORITY && optimalSet.size() != 1) {
            fusionType.setWeights(search.assignWeights(optimalSet));
        }
    }

    /**
     * 
     * @param is
     * @return
     * @throws Exception
     */
    double test(Instance is) throws Exception {
        int size = optimalSet.size();
        double[] votes = new double[size];

        for (int j = 0; j < size; ++j) {
            if (fusionType == FusionType.MAJORITY || fusionType == FusionType.WEIGHTED_MAJORITY) {
            	// Mit csin�l?
                votes[j] = optimalSet.get(j).classifyInstance(is);
            } else {
            	// Mit csin�l?
                votes[j] = optimalSet.get(j).distributionForInstance(is)[1];
            }
        }
        fusionType.setVotes(votes);
        return fusionType.apply();
    }

/**
 *  kiolvassa �s be�ll�tja a file-b�l a datasetet.(egy adathalmaz)
 * @param path Az adathalmaz el�r�si �tja
 * @throws IOException 
 */
    private void loadDataset(String path) throws IOException {
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(path));
        dataset = loader.getDataSet();
    }
    
   
    void validate(String datasetPath, int cvFolds, boolean filterAttributes) throws Exception {
        senStat = new DescriptiveStatistics();
        speStat = new DescriptiveStatistics();
        accStat = new DescriptiveStatistics();
        //bet�ltj�k a kapott adatokat(test.arff f�jlb�l) 
        loadDataset(datasetPath);
        // �sszekeveri az adatokat
        dataset.randomize(new Random());
        
        // Ha igaz a filterattributes, akkor mit csin�l?
        if (filterAttributes) {
            AttributeSelection as = new AttributeSelection();
                as.setEvaluator(new ConsistencySubsetEval());
                as.SelectAttributes(dataset);
                as.reduceDimensionality(dataset);
                ASEvaluation ase = new InfoGainAttributeEval();
                ase.buildEvaluator(dataset);
        }

// A keres�nek be�ll�tjuk az oszt�lyoz�it
        search.setClassifiers(classifiers);
// A keres�nek be�ll�tjuk az adathalmazt
        search.setDataset(dataset);
        // Mit csin�l a numfolds? Ez mondja meg hogy h�ny r�szre kell osztani?
        search.setNumFolds(cvFolds);
// A keres�nek be�ll�tjuk a fusiontypeot
        search.setFusionType(fusionType);
// Otpim�lis oszt�lyoz� list�nak be�ll�tja a select �ltal visszaadott keres�s tipus�t�l f�gg� optim�lis oszt�lyoz� list�t
        optimalSet = search.select();
        for (int i = 0; i < cvFolds; ++i) {
        	// Be�ll�tjuk a tanul�si adathalmazt
            Instances trainDataset = dataset.trainCV(cvFolds, i);
           
            // Az oszt�lyoz�kn�l l�trej�nnek a modellek (betanulnak)
            train(trainDataset);
            // Ez a m�sik adatb�zis (tesztel�s)
            Instances test = dataset.testCV(cvFolds, i);
                
                // Visszaad egy felsorol�st az adathalmaz p�ld�nyaib�l
                Enumeration<Instance> instEnum = test.enumerateInstances();
                double tp = 0;
                double fp = 0;
                double tn = 0;
                double fn = 0;
                // A keres�nek az adathalmaz�ra be�ll�tom a traindataset-et
                search.setDataset(trainDataset);
                if (fusionType == FusionType.WEIGHTED_MAJORITY && optimalSet.size() != 1) {
                    fusionType.setWeights(search.assignWeights(optimalSet));
                }
                
                while (instEnum.hasMoreElements()) {
                	// Kiszedj�k a k�vetkez� elem�t
                    Instance is = instEnum.nextElement();
                    // Megk�rdezz�k, hogy az is-nek mi a class value-ja? Mi volt a val�s oszt�lyc�mke
                    double classValue = is.classValue();
                    // MIt mond az oszt�lyoz�
                    double prediction = test(is);
                    if (prediction > 0.5) {
                        prediction = 1.0;
                    } else {
                        prediction = 0.0;
                    }
                    
                    // ha a val�s oszt�lyc�mke nem egyezik meg az oszt�lyoz� �ltal hozz�rendelt c�mk�vel: l�trej�n az a t�bl�zat
                    if (classValue != prediction) {
                        if (classValue == 0.0) {
                            fp++;
                        } else {
                            fn++;
                        }
                    } else {
                        if (classValue == 1.0) {
                            tp++;
                        } else {
                            tn++;
                        }
                    }
                }
// Az �sszes pozit�v k�z�l h�ny volt val�j�ban pozit�v (teh�t h�ny poz�v lett a class value szerint?)
                double sen = tp / (tp + fn);
                if (!Double.isNaN(sen)) {
                    senStat.addValue(sen);
                }
                
// Specifcit�s: az �sszes negat�v k�z�l h�ny volt igaz�b�l negat�v
                double spe = tn / (fp + tn);
                if (!Double.isNaN(spe)) {
                    speStat.addValue(spe);
                }
                
//Pontoss�g: az �sszes adatot n�zve h�nyat tal�lt el j�l
                double acc = (tp + tn) / (tp + tn + fp + fn);
                if (!Double.isNaN(acc)) {
                    accStat.addValue(acc);
                }
                
                nullValueStatistic.add(new NullValueStatistic(fp+tn,tp+fn,acc,fusionType));
                
                System.out.println("nullas ertek: " + fp+tn);
                System.out.println("egyes ertek: " + tp+fn);
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
    	Date start = new Date();
        nullValueStatistic = new ArrayList<NullValueStatistic>();

        for(int i=0; i<10; i++) {
        	List<Classifier> classifiers = new ArrayList<Classifier>();
            classifiers.add(new ADTree());
            classifiers.add(new IBk(33));
            classifiers.add(new AdaBoostM1());
            classifiers.add(new LibSVM());
            classifiers.add(new NaiveBayes());
            classifiers.add(new RandomTree());
            classifiers.add(new PatternClassifier());
            classifiers.add(new MultilayerPerceptron());
                    
            Search s = Search.ALL;
            for(FusionType fusi : FusionType.values()) {
            	 FusionType ft = fusi;
                 System.out.println(s.name() + " " + ft.name());
                 // Be�ll�tja az oszt�lyoz�kat, a fusion type-ot, a keres�s fajt�j�t �s hogy h�nyszor fusson le.
                 EnsembleTrainer trainer = new EnsembleTrainer(classifiers, ft, s, 10);
                 try {
                	 // a trainer validate fv-�t megh�vom �s �tadom a test.arff f�jl el�r�si �tj�t param�terk�nt; 2 r�szre osztom az adatokat(cvfolds) �s ha true(filterattributes), akkor mit csin�l?
                     // false-al nem mukodik
                	 trainer.validate("C:/Users/Eszter/Desktop/test.arff",2,true);
                     System.out.println(trainer.getAccStat());
                 } catch (Exception e) {
                     e.printStackTrace();
                 }
                
            }
        }
        
        Date end = new Date();
        System.out.println("Start: " + start);
        System.out.println("End: " + end);
        
        PrintWriter writer;
		try {
			writer = new PrintWriter("C:/Users/Eszter/Documents/Suli/Szakdoga/NullValueStat.txt", "UTF-8");
			writer.println("Start: " + start);
			writer.println("End: " + end);
	        for(NullValueStatistic n : nullValueStatistic)  {
	        	System.out.println(n.toString());
	        	writer.println(n.toString());
	        }
			writer.close();
			  
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

    }
}