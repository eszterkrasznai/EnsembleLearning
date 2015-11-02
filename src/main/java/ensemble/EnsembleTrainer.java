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
    	// A keresõnek beállítom az osztályozókat
        search.setClassifiers(classifiers);
        // A keresõnek beállítom a tanulási adathalmazát
        search.setDataset(training);
        // A keresõnek megmondom, hogy hány részre van felosztva az adathalmaz?
        search.setNumFolds(numFolds);
        // A keresõnek beállítom, hogy milyen fusiontype-al dolgozzon
        search.setFusionType(fusionType);
        
        optimalSet = search.select();
        for (Classifier classifier : optimalSet) {
        	//Itt történik a tanulás
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
            	// Mit csinál?
                votes[j] = optimalSet.get(j).classifyInstance(is);
            } else {
            	// Mit csinál?
                votes[j] = optimalSet.get(j).distributionForInstance(is)[1];
            }
        }
        fusionType.setVotes(votes);
        return fusionType.apply();
    }

/**
 *  kiolvassa és beállítja a file-ból a datasetet.(egy adathalmaz)
 * @param path Az adathalmaz elérési útja
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
        //betöltjük a kapott adatokat(test.arff fájlból) 
        loadDataset(datasetPath);
        // Összekeveri az adatokat
        dataset.randomize(new Random());
        
        // Ha igaz a filterattributes, akkor mit csinál?
        if (filterAttributes) {
            AttributeSelection as = new AttributeSelection();
                as.setEvaluator(new ConsistencySubsetEval());
                as.SelectAttributes(dataset);
                as.reduceDimensionality(dataset);
                ASEvaluation ase = new InfoGainAttributeEval();
                ase.buildEvaluator(dataset);
        }

// A keresõnek beállítjuk az osztályozóit
        search.setClassifiers(classifiers);
// A keresõnek beállítjuk az adathalmazt
        search.setDataset(dataset);
        // Mit csinál a numfolds? Ez mondja meg hogy hány részre kell osztani?
        search.setNumFolds(cvFolds);
// A keresõnek beállítjuk a fusiontypeot
        search.setFusionType(fusionType);
// Otpimális osztályozó listának beállítja a select által visszaadott keresés tipusától függõ optimális osztályozó listát
        optimalSet = search.select();
        for (int i = 0; i < cvFolds; ++i) {
        	// Beállítjuk a tanulási adathalmazt
            Instances trainDataset = dataset.trainCV(cvFolds, i);
           
            // Az osztályozóknál létrejönnek a modellek (betanulnak)
            train(trainDataset);
            // Ez a másik adatbázis (tesztelés)
            Instances test = dataset.testCV(cvFolds, i);
                
                // Visszaad egy felsorolást az adathalmaz példányaiból
                Enumeration<Instance> instEnum = test.enumerateInstances();
                double tp = 0;
                double fp = 0;
                double tn = 0;
                double fn = 0;
                // A keresõnek az adathalmazára beállítom a traindataset-et
                search.setDataset(trainDataset);
                if (fusionType == FusionType.WEIGHTED_MAJORITY && optimalSet.size() != 1) {
                    fusionType.setWeights(search.assignWeights(optimalSet));
                }
                
                while (instEnum.hasMoreElements()) {
                	// Kiszedjük a következõ elemét
                    Instance is = instEnum.nextElement();
                    // Megkérdezzük, hogy az is-nek mi a class value-ja? Mi volt a valós osztálycímke
                    double classValue = is.classValue();
                    // MIt mond az osztályozó
                    double prediction = test(is);
                    if (prediction > 0.5) {
                        prediction = 1.0;
                    } else {
                        prediction = 0.0;
                    }
                    
                    // ha a valós osztálycímke nem egyezik meg az osztályozó által hozzárendelt címkével: létrejön az a táblázat
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
// Az összes pozitív közül hány volt valójában pozitív (tehát hány pozív lett a class value szerint?)
                double sen = tp / (tp + fn);
                if (!Double.isNaN(sen)) {
                    senStat.addValue(sen);
                }
                
// Specifcitás: az összes negatív közül hány volt igazából negatív
                double spe = tn / (fp + tn);
                if (!Double.isNaN(spe)) {
                    speStat.addValue(spe);
                }
                
//Pontosság: az összes adatot nézve hányat talált el jól
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
                 // Beállítja az osztályozókat, a fusion type-ot, a keresés fajtáját és hogy hányszor fusson le.
                 EnsembleTrainer trainer = new EnsembleTrainer(classifiers, ft, s, 10);
                 try {
                	 // a trainer validate fv-ét meghívom és átadom a test.arff fájl elérési útját paraméterként; 2 részre osztom az adatokat(cvfolds) és ha true(filterattributes), akkor mit csinál?
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