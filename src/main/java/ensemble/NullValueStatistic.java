package ensemble;

import java.util.List;

public class NullValueStatistic {
	
	private double nullValueCount;
	private double oneValueCount;
	private double mean;
	private FusionType fusionType;
	private List<Double> predictions;

	public NullValueStatistic(double nullValueCount, double oneValueCount, double mean,
			FusionType fusionType, List<Double> predictions) {
		super();
		this.nullValueCount = nullValueCount;
		this.oneValueCount = oneValueCount;
		this.mean = mean;
		this.fusionType = fusionType;
		this.predictions = predictions;
	}
	public double getNullValueCount() {
		return nullValueCount;
	}
	
	public double getoneValueCount(){
		return oneValueCount;
	}
	

	public double getMean() {
		return mean;
	}

	public FusionType getFusionType() {
		return fusionType;
	}
	
	public List<Double> getPredictions() {
		return predictions;
	}
	

	
	
	
	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		sb.append("< null values count: " + nullValueCount +" > ");
		sb.append("< one values count: " + oneValueCount + " > ");
		sb.append("< FusionType: " + fusionType + " > ");
		sb.append("< mean: " + mean  + " > ");
		for (int i = 0; i<predictions.size(); i++){
			sb.append("\n");
			sb.append("< " + i + ". classifier: " + predictions.get(i) + " >");
		}
			
	
		return sb.toString();
		
	}
	
}
