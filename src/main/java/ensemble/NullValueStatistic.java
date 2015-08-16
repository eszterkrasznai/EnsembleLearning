package ensemble;

public class NullValueStatistic {
	
	private double nullValueCount;
	private double mean;
	private FusionType fusionType;

	public NullValueStatistic(double nullValueCount, double mean,
			FusionType fusionType) {
		super();
		this.nullValueCount = nullValueCount;
		this.mean = mean;
		this.fusionType = fusionType;
	}
	public double getNullValueCount() {
		return nullValueCount;
	}

	public double getMean() {
		return mean;
	}

	public FusionType getFusionType() {
		return fusionType;
	}

	@Override
	public String toString() {
		return "< null values count: " + nullValueCount +" > < FusionType: " + fusionType + " > < mean: " + mean  + " >";
	}
	
	
	
}
