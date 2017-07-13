package librec.ranking;

import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.IterativeRecommender;
import librec.util.Randoms;
import librec.util.Strings;

public class AR extends IterativeRecommender {
	public AR(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		this.initByNorm = false;
	}

	protected void initModel() throws Exception {
		super.initModel();
		this.itemCache = this.trainMatrix.columnCache(cacheSpec);
	}

	protected void buildModel() throws Exception {

		for (int iter = 1; iter <= numIters; iter++) {
			this.loss = 0.0;
			int s = 0;

			for (int smax = numUsers * 100; s < smax; s++) {
				int k = 0;
				int v = 0;
				int w = 0;
				
				boolean isItemLikedByNooneOrByAll = false;
				do {
					k = Randoms.uniform(numItems); //
					SparseVector userSet = itemCache.get(k);

					isItemLikedByNooneOrByAll = userSet.getCount() == 0 || userSet.getCount() == numUsers;

					if (!isItemLikedByNooneOrByAll) {
						int[] userList = userSet.getIndex();

						v = userList[Randoms.uniform(userList.length)];

						int[] diff = diff(numUsers, userList);

						w = diff[Randoms.uniform(diff.length)];
					}

				} while (isItemLikedByNooneOrByAll);

				double xvk = predict(v, k);
				double xwk = predict(w, k);
				double xvwk = xvk - xwk;

				double vals = Math.log(g(xvwk));

				this.loss += vals;

				double cmg = g(-xvwk) * 0.5;

				for (int f = 0; f < numFactors; f++) {
					double qkf = this.Q.get(k, f);
					double pvf = this.P.get(v, f);
					double pwf = this.P.get(w, f);

					this.Q.add(k, f, this.lRate * (cmg * (pvf - pwf) - regI * qkf));
					this.P.add(v, f, this.lRate * (cmg * qkf - regU * pvf));
					this.P.add(w, f, this.lRate * (cmg * -qkf - regU * pwf));

					this.loss -= regI * qkf * qkf + regU * pvf * pvf + regU * pwf * pwf;
				}

			}

			this.loss *= 0.5;
			if (isConverged(iter))
				break;
		}
	}

	

	public String toString() {
		return Strings.toString(
				new Object[] { Float.valueOf(binThold), Integer.valueOf(numFactors), Float.valueOf(initLRate),
						Float.valueOf(maxLRate), Float.valueOf(regU), Float.valueOf(regI), Integer.valueOf(numIters) },
				",");
	}
}