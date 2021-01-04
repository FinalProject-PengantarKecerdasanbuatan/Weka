package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.UpdateableClassifier;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.WeightedInstancesHandler;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;


public class MKNN 
  extends SingleClassifierEnhancer
  implements UpdateableClassifier, WeightedInstancesHandler {

  protected Instances m_Train;
    
  protected int m_kNN = -1;

  protected int m_WeightKernel = LINEAR;

  protected boolean m_UseAllK = true;
  
  protected NearestNeighbourSearch m_NNSearch =  new LinearNNSearch();
  
  /** The available kernel weighting methods. */
  public static final int LINEAR       = 0;

  /** a ZeroR model in case no model can be built from the data. */
  protected Classifier m_ZeroR;

  public MKNN() {    
    m_Classifier = new weka.classifiers.trees.DecisionStump();
  }

  public Enumeration<Option> listOptions() {
    
    Vector<Option> newVector = new Vector<Option>(3);
    newVector.addElement(new Option("\tThe neighbour search.\n","A", 0, "-A"));
    newVector.addElement(new Option("\tSet The number of neighbour.\n"+"\t(default all)","K", 1, "-K <number of neighbours>"));
    newVector.addElement(new Option("\tSet the weighting kernel shape to use.\n"+"\t(default 0 = Linear)","U", 1,"-U <number of weighting method>"));
    
    newVector.addAll(Collections.list(super.listOptions()));

    return newVector.elements();
  }


  public void setOptions(String[] options) throws Exception {

    String knnString = Utils.getOption('K', options);
    if (knnString.length() != 0) {
      setKNN(Integer.parseInt(knnString));
    } else {
      setKNN(-1);
    }

    String weightString = Utils.getOption('U', options);
    if (weightString.length() != 0) {
      setWeightingKernel(Integer.parseInt(weightString));
    } else {
      setWeightingKernel(LINEAR);
    }
    
    String nnSearchClass = Utils.getOption('A', options);
    if(nnSearchClass.length() != 0) {
      String nnSearchClassSpec[] = Utils.splitOptions(nnSearchClass);
      if(nnSearchClassSpec.length == 0) { 
        throw new Exception("Invalid NearestNeighbourSearch algorithm "); 
      }
      String className = nnSearchClassSpec[0];
      nnSearchClassSpec[0] = "";

      setNearestNeighbourSearchAlgorithm( (NearestNeighbourSearch)
                  Utils.forName( NearestNeighbourSearch.class, 
                                 className, 
                                 nnSearchClassSpec)
                                        );
    }
    else 
      this.setNearestNeighbourSearchAlgorithm(new LinearNNSearch());

    super.setOptions(options);
  }

  public String [] getOptions() {

    Vector<String> options = new Vector<String>();

    options.add("-U"); options.add("" + getWeightingKernel());
    if ( (getKNN() == 0) && m_UseAllK) {
        options.add("-K"); options.add("-1");
    }
    else {
        options.add("-K"); options.add("" + getKNN());
    }
    options.add("-A");
    options.add(m_NNSearch.getClass().getName()+" "+Utils.joinOptions(m_NNSearch.getOptions()));; 

    Collections.addAll(options, super.getOptions());
    
    return options.toArray(new String[0]);
  }

  public void setKNN(int knn) {

    m_kNN = knn;
    if (knn <= 0) {
      m_kNN = 0;
      m_UseAllK = true;
    } else {
      m_UseAllK = false;
    }
  }

  public int getKNN() {

    return m_kNN;
  }

  public void setWeightingKernel(int kernel) {

    if ((kernel != LINEAR)) {
      return;
    }
    m_WeightKernel = kernel;
  }

  public int getWeightingKernel() {

    return m_WeightKernel;
  }

  public NearestNeighbourSearch getNearestNeighbourSearchAlgorithm() {
    return m_NNSearch;
  }

  public void setNearestNeighbourSearchAlgorithm(NearestNeighbourSearch nearestNeighbourSearchAlgorithm) {
    m_NNSearch = nearestNeighbourSearchAlgorithm;
  }

  public Capabilities getCapabilities() {
    Capabilities      result;
    
    if (m_Classifier != null) {
      result = m_Classifier.getCapabilities();
    } else {
      result = super.getCapabilities();
    }
    
    result.setMinimumNumberInstances(0);
    
    // set dependencies
    for (Capability cap: Capability.values())
      result.enableDependency(cap);
    
    return result;
  }

  public void buildClassifier(Instances instances) throws Exception {

    if (!(m_Classifier instanceof WeightedInstancesHandler)) {
      throw new IllegalArgumentException("Classifier must be a "
					 + "WeightedInstancesHandler!");
    }

    // can classifier handle the data?
    getCapabilities().testWithFail(instances);

    // remove instances with missing class
    instances = new Instances(instances);
    instances.deleteWithMissingClass();
    
    // only class? -> build ZeroR model
    if (instances.numAttributes() == 1) {
      System.err.println(
	  "Cannot build model (only class attribute present in data!), "
	  + "using ZeroR model instead!");
      m_ZeroR = new weka.classifiers.rules.ZeroR();
      m_ZeroR.buildClassifier(instances);
      return;
    }
    else {
      m_ZeroR = null;
    }
    
    m_Train = new Instances(instances, 0, instances.numInstances());

    m_NNSearch.setInstances(m_Train);
  }

  public void updateClassifier(Instance instance) throws Exception {

    if (m_Train == null) {
      throw new Exception("No training instance structure set!");
    }
    else if (m_Train.equalHeaders(instance.dataset()) == false) {
      throw new Exception("Incompatible instance types\n" + m_Train.equalHeadersMsg(instance.dataset()));
    }
    if (!instance.classIsMissing()) {
      m_NNSearch.update(instance);
      m_Train.add(instance);
    }
  }

  public double[] distributionForInstance(Instance instance) throws Exception {
    
    // default model?
    if (m_ZeroR != null) {
      return m_ZeroR.distributionForInstance(instance);
    }
    
    if (m_Train.numInstances() == 0) {
      throw new Exception("No training instances!");
    }
    
    m_NNSearch.addInstanceInfo(instance);
    
    int k = m_Train.numInstances();
    if( (!m_UseAllK && (m_kNN < k)) /*&&
       !(m_WeightKernel==INVERSE ||
         m_WeightKernel==GAUSS)*/ ) {
      k = m_kNN;
    }
    
    Instances neighbours = m_NNSearch.kNearestNeighbours(instance, k);
    double distances[] = m_NNSearch.getDistances();

    if (m_Debug) {
      System.out.println("Test Instance: "+instance);
      System.out.println("For "+k+" kept " + neighbours.numInstances() + " out of " + 
                         m_Train.numInstances() + " instances.");
    }
    
    // Pass the distances through a weighting kernel
    for (int i = 0; i < distances.length; i++) {
      switch (m_WeightKernel) {
        case LINEAR:
          distances[i] = 1.0001 - distances[i];
          break;
      }
    }

    if (m_Debug) {
      System.out.println("Instance Weights");
      for (int i = 0; i < distances.length; i++) {
	System.out.println("" + distances[i]);
      }
    }
    
    // Set the weights on the training data
    double sumOfWeights = 0, newSumOfWeights = 0;
    for (int i = 0; i < distances.length; i++) {
      double weight = distances[i];
      Instance inst = (Instance) neighbours.instance(i);
      sumOfWeights += inst.weight();
      newSumOfWeights += inst.weight() * weight;
      inst.setWeight(inst.weight() * weight);

    }

    // Create a weighted classifier
    m_Classifier.buildClassifier(neighbours);

    if (m_Debug) {
      System.out.println("Classifying test instance: " + instance);
      System.out.println("Built base classifier:\n" 
			 + m_Classifier.toString());
    }

    // Return the classifier's predictions
    return m_Classifier.distributionForInstance(instance);
  }


  public static void main(String [] argv) {
    runClassifier(new MKNN(), argv);
  }
}
