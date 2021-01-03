package weka.classifiers.lazy;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.classifiers.SingleClassifierEnhancer;
import weka.classifiers.UpdateableClassifier;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.RevisionUtils;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;
import weka.core.neighboursearch.LinearNNSearch;
import weka.core.neighboursearch.NearestNeighbourSearch;


public class MKNN 
  extends SingleClassifierEnhancer
  implements UpdateableClassifier, WeightedInstancesHandler {

  /** for serialization. */
  static final long serialVersionUID = 1979797405383665815L;

  /** The training instances used for classification. */
  protected Instances m_Train;
    
  /** The number of neighbours used to select the kernel bandwidth. */
  protected int m_kNN = -1;

  /** The weighting kernel method currently selected. */
  protected int m_WeightKernel = LINEAR;

  /** True if m_kNN should be set to all instances. */
  protected boolean m_UseAllK = true;
  
  /** The nearest neighbour search algorithm to use. 
   * (Default: weka.core.neighboursearch.LinearNNSearch) 
   */
  protected NearestNeighbourSearch m_NNSearch =  new LinearNNSearch();
  
  /** The available kernel weighting methods. */
  public static final int LINEAR       = 0;
  public static final int EPANECHNIKOV = 1;
  public static final int TRICUBE      = 2;  
  public static final int INVERSE      = 3;
  public static final int GAUSS        = 4;
  public static final int CONSTANT     = 5;

  /** a ZeroR model in case no model can be built from the data. */
  protected Classifier m_ZeroR;

  public MKNN() {    
    m_Classifier = new weka.classifiers.trees.DecisionStump();
  }

  protected String defaultClassifierString() {
    
    return "weka.classifiers.trees.DecisionStump";
  }

  public Enumeration<String> enumerateMeasures() {
    return m_NNSearch.enumerateMeasures();
  }
 
  public double getMeasure(String additionalMeasureName) {
    return m_NNSearch.getMeasure(additionalMeasureName);
  }

  public Enumeration<Option> listOptions() {
    
    Vector<Option> newVector = new Vector<Option>(3);
    newVector.addElement(new Option("\tThe nearest neighbour search " +
                                    "algorithm to use " +
                                    "(default: weka.core.neighboursearch.LinearNNSearch).\n",
                                    "A", 0, "-A"));
    newVector.addElement(new Option("\tSet the number of neighbours used to set"
				    +" the kernel bandwidth.\n"
				    +"\t(default all)",
				    "K", 1, "-K <number of neighbours>"));
    newVector.addElement(new Option("\tSet the weighting kernel shape to use."
				    +" 0=Linear, 1=Epanechnikov,\n"
				    +"\t2=Tricube, 3=Inverse, 4=Gaussian.\n"
				    +"\t(default 0 = Linear)",
				    "U", 1,"-U <number of weighting method>"));
    
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
        throw new Exception("Invalid NearestNeighbourSearch algorithm " +
                            "specification string."); 
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
 
  public String KNNTipText() {
    return "How many neighbours are used to determine the width of the "
      + "weighting function (<= 0 means all neighbours).";
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

  public String weightingKernelTipText() {
    return "Determines weighting function. [0 = Linear, 1 = Epnechnikov,"+
	   "2 = Tricube, 3 = Inverse, 4 = Gaussian and 5 = Constant. "+
	   "(default 0 = Linear)].";
  }

  public void setWeightingKernel(int kernel) {

    if ((kernel != LINEAR)
	&& (kernel != EPANECHNIKOV)
	&& (kernel != TRICUBE)
	&& (kernel != INVERSE)
	&& (kernel != GAUSS)
	&& (kernel != CONSTANT)) {
      return;
    }
    m_WeightKernel = kernel;
  }

  public int getWeightingKernel() {

    return m_WeightKernel;
  }

  public String nearestNeighbourSearchAlgorithmTipText() {
    return "The nearest neighbour search algorithm to use (Default: LinearNN).";
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
    
    //IF LinearNN has skipped so much that <k neighbours are remaining.
    if(k>distances.length)
      k = distances.length;

    if (m_Debug) {
      System.out.println("Instance Distances");
      for (int i = 0; i < distances.length; i++) {
	System.out.println("" + distances[i]);
      }
    }

    // Determine the bandwidth
    double bandwidth = distances[k-1];

    // Check for bandwidth zero
    if (bandwidth <= 0) {
      //if the kth distance is zero than give all instances the same weight
      for(int i=0; i < distances.length; i++)
        distances[i] = 1;
    } else {
      // Rescale the distances by the bandwidth
      for (int i = 0; i < distances.length; i++)
        distances[i] = distances[i] / bandwidth;
    }
    
    // Pass the distances through a weighting kernel
    for (int i = 0; i < distances.length; i++) {
      switch (m_WeightKernel) {
        case LINEAR:
          distances[i] = 1.0001 - distances[i];
          break;
        case EPANECHNIKOV:
          distances[i] = 3/4D*(1.0001 - distances[i]*distances[i]);
          break;
        case TRICUBE:
          distances[i] = Math.pow( (1.0001 - Math.pow(distances[i], 3)), 3 );
          break;
        case CONSTANT:
          //System.err.println("using constant kernel");
          distances[i] = 1;
          break;
        case INVERSE:
          distances[i] = 1.0 / (1.0 + distances[i]);
          break;
        case GAUSS:
          distances[i] = Math.exp(-distances[i] * distances[i]);
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
      //weightedTrain.add(newInst);
    }
    
    // Rescale weights
    for (int i = 0; i < neighbours.numInstances(); i++) {
      Instance inst = neighbours.instance(i);
      inst.setWeight(inst.weight() * sumOfWeights / newSumOfWeights);
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

  public String toString() {

    // only ZeroR model?
    if (m_ZeroR != null) {
      StringBuffer buf = new StringBuffer();
      buf.append(this.getClass().getName().replaceAll(".*\\.", "") + "\n");
      buf.append(this.getClass().getName().replaceAll(".*\\.", "").replaceAll(".", "=") + "\n\n");
      buf.append("Warning: No model could be built, hence ZeroR model is used:\n\n");
      buf.append(m_ZeroR.toString());
      return buf.toString();
    }
    
    if (m_Train == null) {
      return "Locally weighted learning: No model built yet.";
    }
    String result = "Locally weighted learning\n"
      + "===========================\n";

    result += "Using classifier: " + m_Classifier.getClass().getName() + "\n";

    switch (m_WeightKernel) {
    case LINEAR:
      result += "Using linear weighting kernels\n";
      break;
    case EPANECHNIKOV:
      result += "Using epanechnikov weighting kernels\n";
      break;
    case TRICUBE:
      result += "Using tricube weighting kernels\n";
      break;
    case INVERSE:
      result += "Using inverse-distance weighting kernels\n";
      break;
    case GAUSS:
      result += "Using gaussian weighting kernels\n";
      break;
    case CONSTANT:
      result += "Using constant weighting kernels\n";
      break;
    }
    result += "Using " + (m_UseAllK ? "all" : "" + m_kNN) + " neighbours";
    return result;
  }

  public String getRevision() {
    return RevisionUtils.extract("$Revision$");
  }
  
  public static void main(String [] argv) {
    runClassifier(new MKNN(), argv);
  }
}
