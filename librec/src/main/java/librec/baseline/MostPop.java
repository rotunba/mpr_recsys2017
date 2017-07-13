package librec.baseline;

import java.util.HashMap;
import java.util.Map;
import librec.data.SparseMatrix;
import librec.intf.Recommender;

public class MostPop extends Recommender {
	private Map<Integer, Integer> userPops;
	private Map<Integer, Integer> itemPops;

	public MostPop(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		setAlgoName("MostPop");
	}

	protected void initModel() {
		if (isRankingPred)
			this.itemPops = new HashMap();
		else
			this.userPops = new HashMap();
	}

	protected double ranking(int u, int j) {
		if (isRankingPred) {
			if (!itemPops.containsKey(j))
				itemPops.put(j, trainMatrix.columnSize(j));

			return itemPops.get(j);
		}
		if (!userPops.containsKey(u))
			userPops.put(u, trainMatrix.rowSize(u));

		return userPops.get(u);

	}
}