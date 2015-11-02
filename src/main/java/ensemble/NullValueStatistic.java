package ensemble;

public class NullValueStatistic {
	
	private double nullValueCount;
	private double oneValueCount;
	private double mean;
	private FusionType fusionType;

	public NullValueStatistic(double nullValueCount,double oneValueCount, double mean,
			FusionType fusionType) {
		super();
		this.nullValueCount = nullValueCount;
		this.oneValueCount = oneValueCount;
		this.mean = mean;
		this.fusionType = fusionType;
	}
	public double getNullValueCount() {
		return nullValueCount;
	}
	
	public double getOneValueCount() {
		return oneValueCount;
	}

	public double getMean() {
		return mean;
	}

	public FusionType getFusionType() {
		return fusionType;
	}

	@Override
	public String toString() {
		return "< null values count: " + nullValueCount +" > < One values count: " + oneValueCount + " > < FusionType: " + fusionType + " > < mean: " + mean  + " >";
	}
	
	
	
}
