package ensemble;

/**
 * Created by IntelliJ IDEA.
 * User: Balint
 * Date: 15.06.11
 * Time: 14:23
 * To change this template use File | Settings | File Templates.
 */
public enum FusionType {

    SOFT_MAX,
    SOFT_MIN,
    SOFT_AVG,
    SOFT_MUL,
    MAJORITY,
    WEIGHTED_MAJORITY;

    private double[] votes;
    private double[] weights;

    public void setVotes(double[] votes) {
        this.votes = votes;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    /**
     * Visszaadja a legnagyobb elemet
     * @return legnagyobb elem
     */
    private double max() {
        double max = Double.MIN_VALUE;
        for (int i = 0; i < votes.length; ++i) {
            if (max < votes[i]) {
                max = votes[i];
            }
        }
        return max;
    }

    /**
     * Visszaadja a legkisebb elemet
     * @return legkisebb elem
     */
    private double min() {
        double min = Double.MAX_VALUE;
        for (int i = 0; i < votes.length; ++i) {
            if (min > votes[i]) {
                min = votes[i];
            }
        }
        return min;
    }

    /**
     * Visszaadja az adott elemek �tlag�t
     * @return �tlag
     */
    private double avg() {
        double avg = 0;
        for (int i = 0; i < votes.length; ++i) {
            avg += votes[i];
        }
        return avg / (double)votes.length;
    }

    /**
     * Visszaadja az adott elemek szorzat�t
     * @return szorzat eredm�nye
     */
    private double mul() {
        double mul = votes[0];
        for (int i = 1; i < votes.length; ++i) {
            mul *= votes[i];
        }
        return mul;
    }

    /**
     * Visszaadja a t�bb szavazattal rendelkez� oszt�lyc�mk�t.
     * @return szavaz�s eredm�nye
     */
    private double majorityVoting() {
        double vote = 0;
        boolean isWeighted = weights != null;
        for (int i = 0; i < votes.length; ++i) {
            double weight;
            if (isWeighted) {
                weight = weights[i];
            }
            else {
                weight = 1;
            }
            vote += votes[i] * weight;
        }
        if (!isWeighted) {
            vote /= (double)votes.length;
        }
        return vote;
    }


    /**
     * ?
     * @return
     */
    public double apply() {
        if (votes == null || votes.length == 0) {
            throw new IllegalArgumentException();
        }
        switch (this) {
            case SOFT_MAX: return max();
            case SOFT_MIN: return min();
            case SOFT_AVG: return avg();
            case SOFT_MUL: return mul();
            case MAJORITY:
            case WEIGHTED_MAJORITY: return majorityVoting();
            default: throw new UnsupportedOperationException();
        }
    }
}
